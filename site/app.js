(() => {
  'use strict';
  const config = window.PROJECT_CONFIG || {};
  const data = window.PROJECT_DATA;
  if (!data) return;
  const navToggle = document.querySelector('#nav-toggle');
  const nav = document.querySelector('#main-nav');
  const closeNav = () => { nav.classList.remove('open'); navToggle.setAttribute('aria-expanded', 'false'); navToggle.setAttribute('aria-label', 'Open navigation'); };
  navToggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  });
  nav.querySelectorAll('a').forEach(a => a.addEventListener('click', closeNav));
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && nav.classList.contains('open')) { closeNav(); navToggle.focus(); } });
  const colors = { Quark: '#75A6E5', Atom: '#53C8CC', Molecule: '#A5CFE0' };
  const familyByName = Object.fromEntries(data.families.map(f => [f.name, f]));
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const number = value => Number(value).toLocaleString('en-US');
  const fixed = value => Number(value).toFixed(2);
  const meanLabel = result => result.displayMean || fixed(result.mean);
  const maxHours = Math.max(...data.families.map(f => f.hours));
  const values = {
    modelCount: data.models.filter(m => m.group === 'ours' && !m.reference).length,
    maxHours, tasks: data.tasks, seeds: data.seeds,
    ftCount: data.models.filter(m => m.ft).length,
    lpCount: data.models.filter(m => m.lp).length
  };
  document.querySelectorAll('[data-value]').forEach(el => { el.textContent = number(values[el.dataset.value]); });
  document.querySelectorAll('[data-collection]').forEach(el => {
    const family = familyByName[el.dataset.collection];
    el.querySelector('div > span').innerHTML = number(family.hours) + ' <small>h</small>';
    el.querySelector('.scale-track i').style.width = (family.hours / maxHours * 100) + '%';
  });
  document.querySelectorAll('[data-headline]').forEach(el => {
    const method = el.dataset.headline;
    const ranked = data.models.filter(m => m[method]).sort((a, b) => b[method].mean - a[method].mean);
    const winner = ranked[0], baseline = ranked.find(m => m.group !== 'ours');
    const leadingOurs = ranked.findIndex(m => m.group !== 'ours');
    const rankWinner = [...ranked].sort((a, b) => a[method].avgRank - b[method].avgRank)[0];
    const score = winner[method].mean, baseScore = baseline[method].mean;
    const bar = (model, isBaseline) => `<div${isBaseline ? ' class="baseline"' : ''}><span>${escape(model.name)}</span><i aria-hidden="true"><b style="width:${model[method].mean / 60 * 100}%"></b></i><strong>${meanLabel(model[method])}</strong></div>`;
    el.innerHTML = `<div class="result-kicker"><span>${method === 'ft' ? 'Full fine-tuning' : 'Linear probing'}</span><span class="small-tag">${escape(winner.name)}</span></div><p class="big-score">${meanLabel(winner[method])}<span>%</span></p><p class="result-delta">+${fixed(score - baseScore)} <span>percentage points over ${escape(baseline.name)}</span></p><div class="small-comparison">${bar(winner, false)}${bar(baseline, true)}</div><p class="result-footer">Top ${leadingOurs} configurations by mean performance are ours.<br>${escape(rankWinner.name)} ${rankWinner.name === winner.name ? 'also leads' : 'leads'} average task rank.</p>`;
  });
  const remoteUrl = value => {
    try { const url = new URL(value); return url.protocol === 'https:' ? url.href : ''; } catch { return ''; }
  };
  const paperUrl = (() => {
    const value = config.paperUrl || 'assets/report.pdf';
    try { const url = new URL(value, document.baseURI); return ['https:', 'http:', 'file:'].includes(url.protocol) ? value : 'assets/report.pdf'; } catch { return 'assets/report.pdf'; }
  })();
  document.querySelectorAll('[data-paper]').forEach(link => {
    link.href = paperUrl + (link.dataset.page ? '#page=' + link.dataset.page : '');
  });
  function releaseLink(container) {
    const key = container.dataset.release === 'github' ? 'githubUrl' : 'huggingFaceUrl';
    const url = remoteUrl(config[key]);
    const item = document.createElement(url ? 'a' : 'button');
    const label = container.dataset.label;
    if (url) {
      item.href = url; item.target = '_blank'; item.rel = 'noopener';
      if (container.dataset.variant === 'button') {
        item.className = 'button resource-button';
        const icon = document.createElement('img');
        icon.src = container.dataset.release === 'github' ? 'assets/icons/github.svg' : 'assets/icons/huggingface.svg';
        icon.alt = ''; icon.width = 20; icon.height = 20;
        if (container.dataset.release !== 'github') icon.className = 'hf-icon';
        const text = document.createElement('span'); text.textContent = label;
        const arrow = document.createElement('span'); arrow.className = 'button-arrow'; arrow.textContent = '↗'; arrow.setAttribute('aria-hidden', 'true');
        item.append(icon, text, arrow);
      } else item.textContent = label + ' ↗';
    }
    else {
      item.type = 'button'; item.disabled = true; item.setAttribute('aria-label', label + ': coming soon');
      if (container.classList.contains('resource-status')) item.textContent = 'Coming soon';
      else { item.textContent = label; const badge = document.createElement('span'); badge.className = 'release-badge'; badge.textContent = 'Coming soon'; item.append(badge); }
    }
    container.replaceChildren(item);
  }
  document.querySelectorAll('[data-release]').forEach(releaseLink);
  const figures = {
    overview: {
      src: 'assets/overview.png',
      label: 'model overview',
      alt: 'Full-FT and LP performance of LaBraM at Scale models and starred Atom versions, compared with the strongest baseline.',
      caption: 'LaBraM at Scale and the strongest baseline in Full-FT and LP.'
    },
    parameters: {
      src: 'assets/parameter_performance.png',
      label: 'encoder size comparison',
      alt: 'Encoder parameters versus Full-FT and LP performance for 29 pretrained configurations. Filled symbols show LaBraM at Scale; open circles show baselines.',
      caption: 'Encoder size and downstream performance. Dashed lines mark the best baseline; the LP axis includes a break.'
    },
    duration: {
      src: 'assets/duration_performance.png',
      label: 'pretraining duration comparison',
      alt: 'Pretraining hours versus Full-FT and LP performance for 29 pretrained configurations. Unreported durations appear in the NR column.',
      caption: 'Pretraining duration and downstream performance. NR means duration not reported.'
    },
    training: {
      src: 'assets/training_transfer.png',
      label: 'training dynamics comparison',
      alt: 'Training loss, Full-FT and selected-readout LP across the same checkpoints for Atom-B, Atom-L, Molecule-B and Molecule-L.',
      caption: 'Lower pretraining loss does not guarantee better transfer. Loss, Full-FT and LP are aligned within each model.'
    }
  };
  const figureTabs = [...document.querySelectorAll('[data-figure]')];
  function selectFigure(key) {
    const figure = figures[key];
    if (!figure) return;
    figureTabs.forEach(button => {
      const selected = button.dataset.figure === key;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    const img = document.querySelector('#result-figure-image');
    img.src = figure.src; img.alt = figure.alt;
    document.querySelector('#figure-panel').setAttribute('aria-labelledby', 'figure-tab-' + key);
    document.querySelector('#result-figure-caption').textContent = figure.caption;
    const imageLink = document.querySelector('#figure-image-link');
    imageLink.href = figure.src;
    imageLink.setAttribute('aria-label', 'Open ' + figure.label + ' figure at full resolution');
    document.querySelector('#figure-open-link').href = figure.src;
  }
  figureTabs.forEach((button, index) => {
    button.addEventListener('click', () => selectFigure(button.dataset.figure));
    button.addEventListener('keydown', event => {
      const target = event.key === 'ArrowRight' ? (index + 1) % figureTabs.length
        : event.key === 'ArrowLeft' ? (index + figureTabs.length - 1) % figureTabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? figureTabs.length - 1 : null;
      if (target === null) return;
      event.preventDefault();
      selectFigure(figureTabs[target].dataset.figure);
      figureTabs[target].focus();
    });
  });
  if (Array.isArray(config.authors) && config.authors.length) {
    const authors = document.querySelector('#authors'); authors.hidden = false;
    for (const author of config.authors) {
      const url = remoteUrl(author.url); const el = document.createElement(url ? 'a' : 'span');
      el.textContent = author.name || ''; if (author.affiliation) el.title = author.affiliation;
      if (author.affiliationId) { const sup = document.createElement('sup'); sup.textContent = author.affiliationId; el.append(sup); }
      if (url) { el.href = url; el.target = '_blank'; el.rel = 'noopener'; }
      authors.append(el);
    }
  }
  document.querySelector('#family-overview').innerHTML = data.families.map(f => `
    <article class="family-card" style="--family:${colors[f.name]}"><h3>${escape(f.name)} <span>${f.models.length} encoders</span></h3><p class="family-hours">${number(f.hours)}<small>hours</small></p><p>${escape(f.description)}</p></article>
  `).join('');
  function renderModels(family = 'all') {
    const models = data.models.filter(m => m.group === 'ours' && !m.reference && (family === 'all' || m.family === family));
    document.querySelector('#model-rows').innerHTML = models.map(m => {
      const modelUrl = remoteUrl(config.modelUrls?.[m.name]);
      const familyUrl = remoteUrl(config.familyModelUrls?.[m.family]);
      const url = modelUrl || familyUrl || remoteUrl(config.huggingFaceUrl);
      return `<tr style="--family:${colors[m.family]}"><td><span class="family-dot" aria-hidden="true"></span>${escape(m.name)}</td><td>${escape(m.params)}</td><td>${number(familyByName[m.family].hours)}</td><td>${meanLabel(m.ft)}%</td><td>${meanLabel(m.lp)}%</td><td>${url ? `<a class="weight-link" href="${escape(url)}" target="_blank" rel="noopener">${modelUrl ? 'View weights' : 'Browse family'} ↗</a>` : '<span class="soon-text">Coming soon</span>'}</td></tr>`;
    }).join('');
    document.querySelector('#catalog-count').textContent = `${models.length} ${models.length === 1 ? 'model' : 'models'}`;
  }
  document.querySelectorAll('[data-family]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-family]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
    renderModels(button.dataset.family);
  }));
  renderModels();
  const state = { method: 'ft', metric: 'mean', expanded: false };
  const taskState = { category: 'all', query: '', showSD: false, sort: 'mean', ascending: false };
  const methodNames = { ft: 'Full fine-tuning', lp: 'Linear probing' };
  function getRanked() {
    const ranked = data.models.filter(m => m[state.method]).map(m => ({ ...m, value: m[state.method][state.metric] }));
    const sign = state.metric === 'mean' ? -1 : 1;
    ranked.sort((a, b) => sign * (a.value - b.value) || a.name.localeCompare(b.name));
    for (let start = 0; start < ranked.length;) {
      let end = start + 1;
      while (end < ranked.length && Math.abs(ranked[end].value - ranked[start].value) < 1e-9) end++;
      const rank = (start + 1 + end) / 2;
      for (let i = start; i < end; i++) ranked[i].rank = rank;
      start = end;
    }
    return ranked;
  }
  function renderBenchmark() {
    const ranked = getRanked(), winner = ranked[0], mean = state.metric === 'mean', shown = state.expanded ? ranked : ranked.slice(0, 8);
    const method = methodNames[state.method], metricName = mean ? 'Mean balanced accuracy' : 'Average task rank';
    document.querySelector('#benchmark-label').textContent = method + ' · ' + metricName;
    document.querySelector('#benchmark-winner').textContent = winner.name;
    document.querySelector('#benchmark-context').textContent = (mean ? 'Highest mean performance' : 'Lowest average task rank') + ' in ' + (state.method === 'ft' ? 'Full-FT' : 'LP') + '.';
    document.querySelector('#benchmark-score').innerHTML = (mean ? meanLabel(winner[state.method]) : fixed(winner.value)) + (mean ? '<span>%</span>' : '');
    document.querySelector('#benchmark-direction').textContent = mean ? 'Higher is better' : 'Lower is better';
    document.querySelector('#score-column').textContent = mean ? 'Mean performance (%)' : 'Average task rank';
    document.querySelector('#ranking-caption').textContent = method + ' ranked by ' + metricName.toLowerCase();
    document.querySelector('#ranking-scale').textContent = mean ? 'Bar scale: 0–60%' : 'Bar scale: 0–' + ranked.length + ' · lower is better';
    document.querySelector('#ranking-rows').innerHTML = shown.map(m => `<tr class="${m.group === 'ours' ? 'ours' : 'baseline'}" style="--family:${colors[m.family] || '#9aa6ae'}"><td>${m.rank}</td><td><span class="model-name">${m.group === 'ours' ? '<span class="family-dot" aria-hidden="true"></span>' : ''}${escape(m.name)}</span>${m.group === 'random' ? '<span class="random-note">Random initialization</span>' : ''}</td><td><div class="rank-value"><div class="rank-track" aria-hidden="true"><span class="rank-bar" style="width:${m.value / (mean ? 60 : ranked.length) * 100}%"></span></div><strong>${mean ? meanLabel(m[state.method]) : fixed(m.value)}${mean ? '%' : ''}</strong></div></td></tr>`).join('');
    document.querySelector('#ranking-count').textContent = `${shown.length} of ${ranked.length} models · ${data.tasks} tasks`;
    const showAll = document.querySelector('#show-all');
    showAll.innerHTML = state.expanded ? 'Show top 8 <span aria-hidden="true">↑</span>' : 'Show all models <span aria-hidden="true">↓</span>';
    showAll.setAttribute('aria-expanded', String(state.expanded));
  }
  document.querySelectorAll('[data-method], [data-task-method]').forEach(button => button.addEventListener('click', () => {
    state.method = button.dataset.method || button.dataset.taskMethod;
    document.querySelectorAll('[data-method], [data-task-method]').forEach(b => {
      const active = (b.dataset.method || b.dataset.taskMethod) === state.method;
      b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active));
    });
    state.expanded = false; renderBenchmark(); renderTaskResults();
  }));
  document.querySelector('#ranking-metric').addEventListener('change', event => { state.metric = event.target.value; renderBenchmark(); });
  document.querySelector('#show-all').addEventListener('click', () => { state.expanded = !state.expanded; renderBenchmark(); });
  renderBenchmark();
  const taskScroller = document.querySelector('#task-table-scroll');
  const taskHead = document.querySelector('#task-table-head');
  const taskBody = document.querySelector('#task-table-body');
  const taskCategories = [...new Set(data.taskList.map(t => t.category))];
  document.querySelector('#task-category').insertAdjacentHTML('beforeend', taskCategories.map(c => `<option value="${escape(c)}">${escape(c)}</option>`).join(''));
  function renderTaskResults() {
    const cohort = data.models.filter(m => m[state.method]);
    const tasks = data.taskList.filter(t => taskState.category === 'all' || t.category === taskState.category);
    const valueFor = (m, key) => key === 'model' ? m.name : key === 'mean' || key === 'avgRank' ? m[state.method][key] : m[state.method].tasks[key].mean;
    const rows = cohort.filter(m => m.name.toLowerCase().includes(taskState.query)).sort((a, b) => {
      const av = valueFor(a, taskState.sort), bv = valueFor(b, taskState.sort);
      const diff = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
      return (taskState.ascending ? diff : -diff) || a.name.localeCompare(b.name);
    });
    const best = Object.fromEntries(['mean', 'avgRank', ...tasks.map(t => t.id)].map(key => [key, (key === 'avgRank' ? Math.min : Math.max)(...cohort.map(m => valueFor(m, key)))]));
    const sortHeader = (key, label, extra = '') => {
      const active = taskState.sort === key, direction = taskState.ascending ? 'ascending' : 'descending';
      return `<th scope="col" ${extra}${active ? ` aria-sort="${direction}"` : ''}><button type="button" data-task-sort="${escape(key)}" title="Sort by ${escape(label)}">${escape(label)}<span class="sort-indicator" aria-hidden="true">${active ? (taskState.ascending ? '↑' : '↓') : '↕'}</span></button></th>`;
    };
    const groups = [];
    for (const t of tasks) {
      if (groups.at(-1)?.category === t.category) groups.at(-1).count++;
      else groups.push({ category: t.category, count: 1 });
    }
    document.querySelectorAll('#task-table colgroup').forEach(group => group.remove());
    document.querySelector('#task-matrix-caption').insertAdjacentHTML('afterend', '<colgroup span="3"></colgroup>' + groups.map(g => `<colgroup span="${g.count}"></colgroup>`).join(''));
    taskHead.innerHTML = `<tr class="task-group-row">${sortHeader('model', 'Model', 'rowspan="2" class="pinned-model" ')}${sortHeader('mean', 'Mean (%)', 'rowspan="2" class="summary-col" ')}${sortHeader('avgRank', 'Avg. rank', 'rowspan="2" class="summary-col" ')}${groups.map(g => `<th scope="colgroup" colspan="${g.count}">${escape(g.category)}</th>`).join('')}</tr><tr class="task-column-row">${tasks.map(t => sortHeader(t.id, t.label)).join('')}</tr>`;
    const numericCell = (m, key, summary = false) => {
      const score = valueFor(m, key), isBest = Math.abs(score - best[key]) < 1e-9;
      const cell = summary ? null : m[state.method].tasks[key];
      const label = summary ? '' : `${m.name}, ${data.taskList.find(t => t.id === key).label}: ${fixed(score)}%; sample SD ${fixed(cell.std)}${cell.readout ? '; readout ' + (cell.readout === 'no_pool' ? 'No-pool' : 'Pooled') : ''}`;
      return `<td${summary ? ' class="summary-col"' : ''}${label ? ` title="${escape(label)}"` : ''}><span class="cell-score${isBest ? ' best-cell' : ''}">${summary && key === 'mean' ? meanLabel(m[state.method]) : fixed(score)}${isBest ? '<span class="sr-only">, best in full comparison</span>' : ''}</span>${!summary && taskState.showSD ? `<span class="cell-sd">± ${fixed(cell.std)}</span>` : ''}</td>`;
    };
    taskBody.innerHTML = rows.map(m => `<tr class="${m.group === 'ours' ? 'ours' : 'baseline'}" style="--family:${colors[m.family] || '#9aa6ae'}"><th scope="row" class="pinned-model"><span class="task-model-name">${m.group === 'ours' ? '<i class="family-dot" aria-hidden="true"></i>' : ''}${escape(m.name)}</span></th>${numericCell(m, 'mean', true)}${numericCell(m, 'avgRank', true)}${tasks.map(t => numericCell(m, t.id)).join('')}</tr>`).join('');
    document.querySelector('#task-table').classList.toggle('show-sd', taskState.showSD);
    document.querySelector('#task-table-count').textContent = `${rows.length} of ${cohort.length} models · ${tasks.length} of ${data.tasks} tasks`;
    document.querySelector('#task-matrix-caption').textContent = `${methodNames[state.method]}: model-by-task balanced accuracy (%)`;
    document.querySelector('#task-empty').hidden = rows.length !== 0;
    const download = document.querySelector('#task-download');
    download.href = `assets/results_${state.method}.csv`;
    download.innerHTML = `Download ${state.method === 'ft' ? 'Full-FT' : 'LP'} CSV <span aria-hidden="true">↓</span>`;
    requestAnimationFrame(updateTaskScrollButtons);
  }
  taskHead.addEventListener('click', event => {
    const button = event.target.closest('[data-task-sort]'); if (!button) return;
    const key = button.dataset.taskSort;
    taskState.ascending = key === taskState.sort ? !taskState.ascending : key === 'model' || key === 'avgRank';
    taskState.sort = key; renderTaskResults();
    [...taskHead.querySelectorAll('[data-task-sort]')].find(b => b.dataset.taskSort === key)?.focus({ preventScroll: true });
  });
  document.querySelector('#task-model-search').addEventListener('input', event => { taskState.query = event.target.value.trim().toLowerCase(); renderTaskResults(); });
  document.querySelector('#task-category').addEventListener('change', event => {
    taskState.category = event.target.value;
    if (!['model', 'mean', 'avgRank'].includes(taskState.sort) && !data.taskList.some(t => t.id === taskState.sort && (taskState.category === 'all' || t.category === taskState.category))) { taskState.sort = 'mean'; taskState.ascending = false; }
    taskScroller.scrollLeft = 0; renderTaskResults();
  });
  document.querySelector('#task-show-sd').addEventListener('change', event => { taskState.showSD = event.target.checked; renderTaskResults(); });
  function updateTaskScrollButtons() {
    document.querySelector('#tasks-scroll-left').disabled = taskScroller.scrollLeft <= 1;
    document.querySelector('#tasks-scroll-right').disabled = taskScroller.scrollLeft + taskScroller.clientWidth >= taskScroller.scrollWidth - 1;
  }
  for (const [id, direction] of [['tasks-scroll-left', -1], ['tasks-scroll-right', 1]]) {
    document.querySelector('#' + id).addEventListener('click', () => taskScroller.scrollBy({ left: direction * Math.max(200, taskScroller.clientWidth - 210), behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' }));
  }
  taskScroller.addEventListener('scroll', updateTaskScrollButtons, { passive: true });
  window.addEventListener('resize', updateTaskScrollButtons);
  renderTaskResults();
  if (typeof config.citationBibtex === 'string' && config.citationBibtex.trim()) {
    document.querySelector('#citation').hidden = false;
    document.querySelector('#bibtex').textContent = config.citationBibtex;
    document.querySelector('#copy-citation').addEventListener('click', async () => {
      const button = document.querySelector('#copy-citation'), status = document.querySelector('#copy-status');
      try { await navigator.clipboard.writeText(config.citationBibtex); button.textContent = 'Copied'; status.textContent = 'BibTeX copied to clipboard.'; }
      catch { const range = document.createRange(); range.selectNodeContents(document.querySelector('#bibtex')); window.getSelection().removeAllRanges(); window.getSelection().addRange(range); button.textContent = 'Select and copy'; status.textContent = 'Citation selected. Press your usual copy shortcut.'; }
    });
  }
})();

