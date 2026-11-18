from contextlib import asynccontextmanager
import hmac
import os
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import psycopg
import redis

from .service_store import Conflict, NotFound, RecommendationStore


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Recommendation(StrictModel):
    request_id: str = Field(pattern=r'^[0-9a-fA-F-]{36}$')
    user_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    at: int = Field(ge=0, le=4102444800)
    history: list[str] = Field(default_factory=list, max_length=20)
    policy: Literal['relevance', 'fresh', 'balanced'] = 'relevance'
    limit: int = Field(default=10, ge=1, le=20)


class Feedback(StrictModel):
    feedback_id: str = Field(pattern=r'^[0-9a-fA-F-]{36}$')
    request_id: str = Field(pattern=r'^[0-9a-fA-F-]{36}$')
    user_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    article_id: str = Field(min_length=1, max_length=100)
    kind: Literal['click', 'dismiss']
    observed_at: int = Field(ge=0, le=4102444800)


class Article(StrictModel):
    article_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    title: str = Field(min_length=1, max_length=500)
    category: str = Field(min_length=1, max_length=100)
    subcategory: str = Field(default='', max_length=100)
    first_seen: int = Field(ge=0, le=4102444800)


class ArticleBatch(StrictModel):
    articles: list[Article] = Field(min_length=1, max_length=256)


def create_app(store=None, engine=None, cache=None, token=None):
    from .service_engine import RecommendationCache, RecommendationEngine
    token = token or os.environ.get('NEWS_API_TOKEN', '')
    if len(token) < 24:
        raise ValueError('Set a private NEWS_API_TOKEN of at least24 characters')
    store = store or RecommendationStore(os.environ['NEWS_DATABASE_URL'])
    engine = engine or RecommendationEngine(os.environ['NEWS_MODEL_MANIFEST'])
    if cache is None:
        url = os.environ.get('NEWS_REDIS_URL')
        client = redis.Redis.from_url(url, socket_connect_timeout=.25, socket_timeout=.25) if url else None
        cache = RecommendationCache(client)

    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
        yield
        if cache.client is not None:
            cache.client.close()

    app = FastAPI(title='News Recommendation Research Service', version='0.1.0', lifespan=lifespan,
                  description='Local authenticated replay service. Caller supplies the replay clock and longer-term history; this is not a public production identity system.')

    def authorize(authorization: str = Header(default='')):
        if not hmac.compare_digest(authorization, 'Bearer ' + token):
            raise HTTPException(401, 'Authentication required')

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={'detail': str(error)})

    @app.exception_handler(NotFound)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={'detail': str(error)})

    @app.exception_handler(psycopg.Error)
    async def store_failure(request, error):
        code = 409 if isinstance(error, psycopg.errors.UniqueViolation) else 503
        return JSONResponse(status_code=code, content={'detail': 'Conflicting identifier' if code == 409 else 'Durable store unavailable'})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={'detail': str(error)})

    @app.get('/health', dependencies=[Depends(authorize)])
    def health():
        with store.connect() as con:
            revision = con.execute('SELECT revision FROM newsrec.catalog_state WHERE singleton=1').fetchone()['revision']
        available = False
        if cache.client is not None:
            try:
                available = bool(cache.client.ping())
            except redis.RedisError:
                pass
        if not revision:
            raise HTTPException(503, 'Catalog has not been initialized')
        return {'status': 'ready', 'cache_available': available, 'catalog_revision': revision, 'model': engine.fingerprint,
                'index_revision': getattr(engine, 'generation', None),
                'index_articles': getattr(getattr(engine, 'index', None), 'index', None).ntotal if hasattr(engine, 'index') else 0}

    @app.post('/recommendations', dependencies=[Depends(authorize)])
    def recommend(body: Recommendation):
        payload = body.model_dump()
        payload['request_id'] = str(UUID(body.request_id))
        if any(not item or len(item) > 100 for item in body.history):
            raise ValueError('Invalid history article ID')
        return store.recommend(payload, engine, cache)

    @app.post('/feedback', dependencies=[Depends(authorize)])
    def feedback(body: Feedback):
        payload = body.model_dump()
        for field in ('feedback_id', 'request_id'):
            payload[field] = str(UUID(payload[field]))
        return store.feedback(payload)

    @app.post('/articles', dependencies=[Depends(authorize)])
    def articles(body: ArticleBatch):
        records = [item.model_dump() for item in body.articles]
        if len({a['article_id'] for a in records}) != len(records):
            raise ValueError('Article IDs in a batch must be unique')
        vectors = engine.encode_articles(records)
        return store.add_articles([{**a, 'fit_clicks': 0, 'vector': v} for a, v in zip(records, vectors)])

    return app
