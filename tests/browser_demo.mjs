// Existing Node's built-in fetch/WebSocket only; no npm packages or downloads.
import {writeFileSync} from 'node:fs';
const [port, url, output] = process.argv.slice(2);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let targets;
for (let i = 0; i < 100; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); }
  catch { await sleep(100); continue; }
  if (targets.some(t => t.type === 'page')) break;
  await sleep(100);
}
const target = targets?.find(t => t.type === 'page');
if (!target) throw new Error('No isolated test-browser target');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let next = 0;
const pending = new Map();
ws.onmessage = event => {
  const data = JSON.parse(event.data);
  if (!data.id || !pending.has(data.id)) return;
  const [resolve, reject] = pending.get(data.id); pending.delete(data.id);
  if (data.error) reject(new Error(data.error.message)); else resolve(data.result);
};
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++next; pending.set(id, [resolve, reject]); ws.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
  if (result.exceptionDetails) throw new Error('Browser JavaScript exception');
  return result.result.value;
}
async function until(expression) {
  for (let i = 0; i < 100; i++) { if (await evaluate(expression)) return; await sleep(100); }
  throw new Error(`UI did not settle: ${await evaluate('document.body.innerText')}`);
}
const settled = `document.getElementById('saved-status') && (document.getElementById('saved-status').textContent.includes('saved development replay') || document.getElementById('saved-status').textContent.includes('Saved comparison unavailable: load'))`;
async function screenshot(suffix) {
  const shot = await send('Page.captureScreenshot', {format:'png', captureBeyondViewport:true});
  writeFileSync(output.slice(0, -5) + suffix + '.png', Buffer.from(shot.data, 'base64'));
}
async function theme(name) {
  if (await evaluate('document.documentElement.dataset.theme') !== name) {
    await evaluate(`document.getElementById('theme-toggle').click()`);
  }
  await until(`document.documentElement.dataset.theme === '${name}'`);
}
const metrics = `(() => {
  const root = getComputedStyle(document.documentElement);
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 1;
  const ctx = canvas.getContext('2d');
  function luminance(token) {
    ctx.clearRect(0,0,1,1); ctx.fillStyle = root.getPropertyValue(token); ctx.fillRect(0,0,1,1);
    const rgb = Array.from(ctx.getImageData(0,0,1,1).data).slice(0,3).map(c => c / 255);
    return rgb.map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4).reduce((sum,c,i) => sum + c * [.2126,.7152,.0722][i], 0);
  }
  const pairs = [['--color-ink','--color-surface'], ['--color-muted','--color-paper'], ['--color-secondary','--color-paper'], ['--color-accent','--color-accent-soft'], ['--color-accent-ink','--color-accent']];
  const contrast = pairs.map(([a,b]) => { const x=luminance(a), y=luminance(b); return (Math.max(x,y)+.05)/(Math.min(x,y)+.05); });
  const wrapped = [];
  document.querySelectorAll('button,nav a').forEach(el => {
    if (!el.getClientRects().length) return;
    for (const node of el.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE || !node.textContent.trim()) continue;
      const range = document.createRange(); range.selectNodeContents(node);
      if (range.getClientRects().length > 1) wrapped.push(el.id || el.className);
    }
  });
  return {mode:document.body.dataset.mode, theme:document.documentElement.dataset.theme, width:innerWidth, scroll:Math.max(document.documentElement.scrollWidth,document.body.scrollWidth), wrapped, contrast, paper:root.getPropertyValue('--color-paper').trim(), themeLabel:document.getElementById('theme-toggle').getAttribute('aria-label')};
})()`;
try {
  await send('Emulation.setDeviceMetricsOverride', {width:1280, height:900, deviceScaleFactor:1, mobile:false});
  await send('Emulation.setEmulatedMedia', {features:[{name:'prefers-color-scheme',value:'dark'}]});
  await send('Page.navigate', {url});
  await until(settled);
  const systemDefault = await evaluate('document.documentElement.dataset.theme');
  await theme('light');
  await send('Page.reload'); await until(settled);
  const themePreserved = await evaluate(`document.documentElement.dataset.theme === 'light'`);
  const views = [];
  for (const name of ['light', 'dark']) {
    await theme(name);
    for (const width of [320,375,414,768,1280]) {
      await send('Emulation.setDeviceMetricsOverride', {width, height:900, deviceScaleFactor:1, mobile:false});
      await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
      views.push(await evaluate(metrics));
      if (width === 375 || width === 1280) await screenshot('-' + name + '-' + width);
    }
  }
  const saved = await evaluate('document.documentElement.outerHTML');
  await screenshot('');
  await evaluate(`location.hash='#live'; document.getElementById('query').value='red apple'; document.getElementById('live-form').requestSubmit();`);
  await until(`document.getElementById('live-status').textContent.includes('products returned')`);
  const live = await evaluate('document.documentElement.outerHTML');
  for (const name of ['light', 'dark']) {
    await theme(name);
    for (const width of [320,375,414,768,1280]) {
      await send('Emulation.setDeviceMetricsOverride', {width, height:900, deviceScaleFactor:1, mobile:false});
      await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
      views.push(await evaluate(metrics));
      if (width === 375 || width === 1280) await screenshot('-live-' + name + '-' + width);
    }
  }
  await send('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  const reducedMotion = await evaluate(`matchMedia('(prefers-reduced-motion: reduce)').matches && getComputedStyle(document.body).animationName === 'none'`);
  await evaluate(`
    window.nativeDemoFetch = window.fetch;
    window.fetch = (path, options) => {
      if (path === '/search') {
        const query = JSON.parse(options.body).query;
        if (query === 'wireless headphones') return new Promise((resolve, reject) => { window.rejectOldSearch = () => reject(new Error('older request failed')); });
        if (query === 'running shoes') return new Promise(resolve => { window.releaseNewSearch = () => resolve(window.nativeDemoFetch(path, options)); });
      }
      return window.nativeDemoFetch(path, options);
    };
    document.querySelector('[data-query="wireless headphones"]').click();
    document.querySelector('[data-query="running shoes"]').click();
    window.rejectOldSearch();
  `);
  await sleep(20);
  const pendingRaceSafe = await evaluate(`document.getElementById('search-button').disabled && document.getElementById('live-status').textContent === 'Searching BM25…'`);
  await evaluate('window.releaseNewSearch()');
  await until(`document.getElementById('live-status').textContent.includes('products returned')`);
  await evaluate(`
    window.fetch = (path, options) => {
      if (path === '/search' && JSON.parse(options.body).query === 'wireless headphones') return new Promise((resolve, reject) => { window.rejectOlderSearch = () => reject(new Error('late old failure')); });
      return window.nativeDemoFetch(path, options);
    };
    document.querySelector('[data-query="wireless headphones"]').click();
    document.getElementById('query').value = 'red apple';
    document.getElementById('live-form').requestSubmit();
  `);
  await until(`document.getElementById('live-status').textContent.includes('products returned')`);
  await evaluate('window.rejectOlderSearch()'); await sleep(20);
  const completedRaceSafe = await evaluate(`document.getElementById('live-status').textContent.includes('products returned') && !document.getElementById('search-button').disabled`);
  await evaluate('window.fetch = window.nativeDemoFetch');
  writeFileSync(output, JSON.stringify({saved, live, views, systemDefault, themePreserved, reducedMotion, pendingRaceSafe, completedRaceSafe}));
} finally { ws.close(); }
