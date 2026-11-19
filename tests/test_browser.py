import os
from pathlib import Path
import socket
import sys
import threading
import time

import pytest

pytest.importorskip('playwright.sync_api')
pytest.importorskip('uvicorn')
pytest.importorskip('fastapi')
if os.environ.get('NEWS_BROWSER_TESTS')!='1':pytest.skip('Set NEWS_BROWSER_TESTS=1 with a browser installed',allow_module_level=True)
from playwright.sync_api import expect, sync_playwright
import uvicorn


def test_browser_feedback_policy_keyboard_mobile_and_comparison(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from examples.preview import create_preview
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(create_preview(),host='127.0.0.1',port=port,log_level='error'))
    thread=threading.Thread(target=server.run);thread.start()
    errors=[]
    try:
        for _ in range(100):
            if server.started:break
            time.sleep(.05)
        assert server.started
        with sync_playwright() as p:
            options={'headless':True}
            if os.environ.get('NEWS_BROWSER_CHANNEL'):options['channel']=os.environ['NEWS_BROWSER_CHANNEL']
            browser=p.chromium.launch(**options)
            page=browser.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{port}',wait_until='networkidle')
            expect(page.locator('.story')).to_have_count(10)
            expect(page.locator('.research-label')).to_have_text('Authored UI preview')
            expect(page.locator('#policy-description')).to_contain_text('no model inference')
            for policy in ('fresh','balanced','relevance'):
                page.locator(f'input[value={policy}]').check()
                expect(page.locator('#feed')).to_have_attribute('aria-busy','false')
                expect(page.locator(f'input[value={policy}]')).to_be_checked()
            page.locator('.story .feedback').first.click()
            expect(page.locator('.story .feedback').first).to_have_text('Interest saved')
            expect(page.locator('.read-summary').first).to_be_focused()
            page.locator('#refresh').click()
            expect(page.locator('#clock')).to_have_text('14 Nov 2019 · 12:06')
            expect(page.locator('#feed-meta')).to_contain_text('1 recent feedback group')
            page.locator('.read-summary').first.click()
            expect(page.locator('#story-dialog')).to_be_visible()
            expect(page.locator('#dialog-details dd')).to_have_count(4)
            page.keyboard.press('Escape');expect(page.locator('#story-dialog')).not_to_be_visible()
            title=page.locator('.story h3').first.inner_text()
            page.locator('.story').first.get_by_role('button',name='Less like this:',exact=False).click()
            page.locator('#refresh').click();expect(page.locator('.story')).to_have_count(9)
            assert title not in page.locator('.story h3').all_inner_texts()
            page.get_by_role('button',name='Sports',exact=True).click();expect(page.locator('#history li')).to_have_count(2)
            page.locator('#results-tab').click();expect(page.locator('#comparison-rows tr')).to_have_count(3)
            expect(page.locator('#comparison-note')).to_contain_text('not experimental results')
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.locator('#feed-tab').click();page.locator('#reset').click();expect(page.locator('#history li')).to_have_count(1)
            expect(page.locator('.story')).to_have_count(10)
            page.locator('.read-summary').first.click();expect(page.locator('#story-dialog')).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert errors==[]
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=15);assert not thread.is_alive()
