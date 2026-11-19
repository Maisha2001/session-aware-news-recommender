"""Authored UI preview. No benchmark data, learned inference or model evidence."""
from pathlib import Path
import threading

from fastapi.responses import HTMLResponse, Response
from starlette.routing import Route

from newsrec.demo import REPLAY_START, create_demo


STORIES=[
    ('news','A quieter commute starts with better bus connections','An authored example explores how frequent connections could make a neighborhood easier to reach. This is illustrative content, not a published news story.'),
    ('science','What a small urban wetland can teach a growing city','An authored science summary follows a fictional field team mapping seasonal changes in a neighborhood wetland.'),
    ('lifestyle','Inside the repair café giving old radios another chapter','An authored community story imagines volunteers sharing practical repair skills at a weekend workshop.'),
    ('sports','A community team finds its rhythm after a difficult start','An authored sports example describes a fictional team learning to communicate through practice.'),
    ('finance','The local bookshop experimenting with a shared delivery route','An authored business example considers how neighboring shops might coordinate a small delivery service.'),
    ('travel','A slower journey along an old riverside walking route','An authored travel sketch visits imagined footpaths, viewpoints and rest stops along a river.'),
    ('science','Students turn rainfall measurements into a neighborhood map','An authored science example shows a fictional classroom collecting and comparing simple observations.'),
    ('lifestyle','A kitchen garden designed around a few everyday meals','An authored lifestyle summary imagines a small garden planned around familiar seasonal ingredients.'),
    ('sports','Why the warm-up became the most useful part of practice','An authored sports example follows a fictional coach revising a team routine.'),
    ('news','A library makes room for a new kind of community workshop','An authored civic example describes an imagined library hosting practical skill-sharing sessions.')]
CATALOG={f'example-{i}':{'category':category,'title':title,'abstract':summary} for i,(category,title,summary) in enumerate(STORIES)}
PROFILES=[{'id':'discover','label':'Surprise me','history':[],'description':'Authored empty-history example'},
          {'id':'science','label':'Science','history':['example-1','example-6'],'description':'Two authored starting interests'},
          {'id':'sports','label':'Sports','history':['example-3','example-8'],'description':'Two authored starting interests'}]
REPORTS={'example':{'label':'Illustrative numbers only','impressions':12,
    'rows':[{'name':name,'logged_auc':auc,'logged_ndcg10':ndcg,'catalog_recall100':recall}
            for name,auc,ndcg,recall in [('Example baseline',.50,.25,.03),('Example retrieval',.60,.35,.02),('Example ranking',.62,.36,.04)]],
    'note':'Authored UI fixtures only. These numbers are not experimental results. Use the real demo and published aggregate reports for measured evidence.'}}


class PreviewBackend:
    def __init__(self):
        self.lock=threading.Lock();self.readers={};self.receipts={};self.feedback=set()

    def post(self,path,body):
        with self.lock:
            reader=self.readers.setdefault(body['user_id'],{'clicks':[],'dismissed':set()})
            if path=='/feedback':
                duplicate=body['feedback_id'] in self.feedback
                if not duplicate:
                    self.feedback.add(body['feedback_id'])
                    if body['kind']=='dismiss':reader['dismissed'].add(body['article_id'])
                    else:reader['clicks'].append(body['observed_at'])
                return {'duplicate':duplicate,'accepted':True}
            self.receipts[body['request_id']]=body
            order=[a for a in CATALOG if a not in reader['dismissed']]
            if body['policy']=='fresh':order.reverse()
            if body['policy']=='balanced':order=order[::2]+order[1::2]
            groups=sum(when+300<body['at'] for when in reader['clicks'])
            return {'request_id':body['request_id'],'session_groups':groups,'candidate_count':len(order),
                    'items':[{'article_id':a,'title':CATALOG[a]['title'],'category':CATALOG[a]['category'],
                              'score':1.0-i*.05,'history_similarity':.2 if body['history'] else 0.,
                              'session_similarity':.1 if groups else 0.,'first_seen_age_hours':float(6+i)} for i,a in enumerate(order[:10])]}


def create_preview():
    app=create_demo(PreviewBackend(),CATALOG,PROFILES,REPORTS)
    from newsrec import demo
    source=Path(demo.__file__).with_name('web')/'index.html'
    html=source.read_text(encoding='utf-8').replace('Historical replay','Authored UI preview')
    html=html.replace('A research demo using licensed MIND news content. Recommendations are experimental. Feedback changes the replayed context; it does not retrain the model.',
                      'Authored UI preview only: invented stories, deterministic display rules and illustrative numbers. No trained model or benchmark data is used. See the real local demo for model inference.')
    html=html.replace("Order by the frozen model's ranking score.","Illustrative ordering only; no model inference.")
    async def preview_index(request):return HTMLResponse(html)
    script=source.with_name('app.js').read_text(encoding='utf-8')
    for original,replacement in {
        "Order by the frozen model's ranking score.":"Illustrative ordering only; no model inference.",
        "Trade some relevance for more recently observed stories.":"Reverse the authored order to demonstrate this control.",
        "Balance relevance, first-observation age and content/category variety.":"Alternate authored positions to demonstrate this control.",
        "Ranking score":"Illustrative display score",
        "Starting-interest similarity":"Illustrative interest value",
        "Recent-choice similarity":"Illustrative feedback value",
    }.items():script=script.replace(original,replacement)
    async def preview_script(request):return Response(script,media_type='text/javascript')
    app.router.routes.insert(0,Route('/assets/app.js',preview_script,methods=['GET']))
    app.router.routes.insert(0,Route('/',preview_index,methods=['GET']))
    return app
