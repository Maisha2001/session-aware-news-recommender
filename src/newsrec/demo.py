from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Literal
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import httpx

from .data import connect_readonly, file_sha256

REPLAY_START = int(datetime(2019, 11, 14, 12, tzinfo=timezone.utc).timestamp())
REPLAY_END = int(datetime(2019, 11, 14, 23, 59, tzinfo=timezone.utc).timestamp())


class ServiceClient:
    def __init__(self, url, token):
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
            raise ValueError('Demo backend must be a local HTTP service')
        self.client = httpx.Client(base_url=url, headers={'Authorization': 'Bearer ' + token}, timeout=35)

    def post(self, path, body):
        try:
            response = self.client.post(path, json=body)
        except httpx.HTTPError as error:
            raise HTTPException(503, 'The local recommendation service is unavailable. Start it and retry.') from error
        if response.status_code != 200:
            raise HTTPException(503 if response.status_code >= 500 else 409, 'The recommendation service could not complete this action. Refresh the session and retry.')
        return response.json()


def load_catalog(database, manifest):
    database = Path(database)
    if file_sha256(database) != json.loads(Path(manifest).read_text())['source_database_sha256']:
        raise ValueError('Demo requires the original training-archive catalog')
    with connect_readonly(database) as con:
        rows = con.execute('''SELECT a.news_id,a.title,a.abstract,a.category,o.first_seen
                              FROM articles a JOIN observations o USING(news_id)
                              WHERE o.first_seen<'2019-11-14T12:00:00' ''').fetchall()
    catalog = {r[0]: {'title': r[1], 'abstract': r[2], 'category': r[3]} for r in rows}
    groups = {}
    for article, value in catalog.items():
        groups.setdefault(value['category'], []).append(article)
    names = {'news': 'World & current affairs', 'sports': 'Sports', 'finance': 'Business', 'lifestyle': 'Life & culture', 'health': 'Health', 'travel': 'Travel'}
    profiles = [{'id': 'discover', 'label': 'Surprise me', 'history': [], 'description': 'Start with no selected interests.'}]
    for category, label in names.items():
        articles = sorted(groups.get(category, []), key=lambda a: hashlib.sha256(('news-demo-profile-v1:' + a).encode()).hexdigest())[:5]
        if articles:
            profiles.append({'id': category, 'label': label, 'history': articles, 'description': 'Five previously available stories set a starting interest; this is an authored profile.'})
    return catalog, profiles


def comparisons(directory):
    root = Path(directory)
    baseline = json.loads((root / 'baseline-v1/report.json').read_text())
    retrieval = json.loads((root / 'retrieval-v1/report.json').read_text())
    ranking = json.loads((root / 'session-v1/report.json').read_text())
    def row(name, logged, catalog):
        return {'name': name, 'logged_auc': logged['auc']['mean'], 'logged_ndcg10': logged['ndcg@10']['mean'],
                'catalog_recall100': catalog['recall@100']['mean']}
    values = [row('Popularity', baseline['impression_ranking']['popularity']['all'], baseline['catalog_retrieval']['all']),
              row('Content retrieval', retrieval['arms']['learned_content']['logged_ranking']['all'], retrieval['arms']['learned_content']['catalog_retrieval']['all']),
              row('Session ranking', ranking['logged_ranking']['session']['all'], ranking['catalog_ranking']['relevance']['all'])]
    result = {'development': {'label': 'Chronological development', 'impressions': baseline['counts']['impressions'], 'rows': values,
                              'note': 'November13–14 from the training archive. Catalog recall measures known clicked articles; unexposed relevance is unknown. The session ranker uses a learned+popular candidate union.'}}
    final = root / 'heldout-v1/report.json'
    if final.exists():
        data = json.loads(final.read_text())
        result['final'] = {'label': 'Final chronological holdout', 'impressions': data['counts']['impressions'], 'rows': data['comparison_rows'], 'note': data['comparison_note']}
    return result


class DemoSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.csrf = secrets.token_urlsafe(24)
        self.reset()

    def reset(self):
        self.user = 'demo_' + uuid4().hex
        self.at, self.profile, self.policy = REPLAY_START, 'discover', 'relevance'
        self.receipt, self.activity, self.marked = None, [], {}


class Action(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['refresh', 'profile', 'policy', 'feedback', 'reset']
    value: str = Field(default='', max_length=100)
    article_id: str = Field(default='', max_length=100)


def create_demo(client=None, catalog=None, profiles=None, reports=None):
    if catalog is None:
        catalog, profiles = load_catalog(os.environ['NEWS_DEMO_DATABASE'], os.environ['NEWS_MODEL_MANIFEST'])
    if reports is None:
        reports = comparisons(os.environ['NEWS_RESULTS_DIRECTORY'])
    client = client or ServiceClient(os.environ.get('NEWS_SERVICE_URL', 'http://127.0.0.1:8015'), os.environ['NEWS_API_TOKEN'])
    profiles_by_id = {p['id']: p for p in profiles}
    sessions, allocation = OrderedDict(), threading.Lock()
    web = Path(__file__).with_name('web')

    @asynccontextmanager
    async def lifespan(app):
        yield
        if isinstance(client, ServiceClient):
            client.client.close()

    app = FastAPI(title='Fieldnotes', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def browser_headers(request, call_next):
        origin = request.headers.get('origin')
        if request.method == 'POST' and origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Use this local page to send feedback.'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        return response

    def session(request, response, create=False):
        cookie = request.cookies.get('news_demo')
        with allocation:
            value = sessions.get(cookie)
            if value is None:
                if not create:
                    raise HTTPException(409, 'This demo session expired. Reload the page.')
                cookie = secrets.token_urlsafe(24)
                value = sessions[cookie] = DemoSession()
                while len(sessions) > 128:
                    sessions.popitem(last=False)
                response.set_cookie('news_demo', cookie, httponly=True, samesite='strict', max_age=3600)
            sessions.move_to_end(cookie)
        return value

    def recommend(value):
        profile = profiles_by_id[value.profile]
        value.receipt = client.post('/recommendations', {'request_id': str(uuid4()), 'user_id': value.user, 'at': value.at,
                                                        'history': profile['history'], 'policy': value.policy, 'limit': 10})
        value.marked = {}

    def snapshot(value):
        items = [{**item, 'abstract': catalog.get(item['article_id'], {}).get('abstract') or 'A summary was not supplied for this story.',
                  'feedback': value.marked.get(item['article_id'])} for item in value.receipt['items']]
        return {'csrf': value.csrf, 'profile': value.profile, 'policy': value.policy,
                'at': value.at, 'time_label': datetime.fromtimestamp(value.at, timezone.utc).strftime('%d %b %Y · %H:%M'),
                'can_advance': value.at + 360 <= REPLAY_END, 'items': items, 'activity': value.activity[-8:][::-1],
                'session_groups': value.receipt['session_groups'], 'candidate_count': value.receipt['candidate_count'],
                'profiles': [{k: p[k] for k in ('id', 'label', 'description')} for p in profiles],
                'history': [{'title': catalog[a]['title'], 'category': catalog[a]['category']} for a in profiles_by_id[value.profile]['history']]}

    @app.get('/')
    def index():
        return FileResponse(web / 'index.html')

    @app.get('/api/session')
    def bootstrap(request: Request, response: Response):
        value = session(request, response, create=True)
        with value.lock:
            if value.receipt is None:
                recommend(value)
            return snapshot(value)

    @app.get('/api/comparisons')
    def models():
        return reports

    @app.post('/api/action')
    def action(body: Action, request: Request, response: Response):
        value = session(request, response)
        if not hmac.compare_digest(request.headers.get('x-news-demo', ''), value.csrf):
            raise HTTPException(403, 'Reload this local page before sending an action.')
        with value.lock:
            if body.action == 'reset':
                value.reset()
            elif body.action == 'profile':
                if body.value not in profiles_by_id:
                    raise HTTPException(422, 'Unknown interest profile')
                value.reset()
                value.profile = body.value
            elif body.action == 'policy':
                if body.value not in ('relevance', 'fresh', 'balanced'):
                    raise HTTPException(422, 'Unknown ranking preference')
                value.policy = body.value
            elif body.action == 'refresh':
                if value.at + 360 > REPLAY_END:
                    raise HTTPException(409, 'End of this replay day. Start a new session.')
                value.at += 360
            else:
                if body.value not in ('click', 'dismiss'):
                    raise HTTPException(422, 'Unknown feedback')
                item = next((x for x in value.receipt['items'] if x['article_id'] == body.article_id), None)
                if item is None:
                    raise HTTPException(422, 'Choose a story in the current feed')
                identity = str(uuid5(NAMESPACE_URL, value.receipt['request_id'] + ':' + body.article_id + ':' + body.value))
                accepted = client.post('/feedback', {'feedback_id': identity, 'request_id': value.receipt['request_id'], 'user_id': value.user,
                                                    'article_id': body.article_id, 'kind': body.value, 'observed_at': value.at})
                value.marked[body.article_id] = body.value
                if not accepted['duplicate']:
                    value.activity.append({'title': item['title'], 'kind': body.value})
                # Feedback does not manufacture a later impression or skip the fixed delay.
                return snapshot(value)
            recommend(value)
            return snapshot(value)

    app.mount('/assets', StaticFiles(directory=web), name='assets')
    return app
