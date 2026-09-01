const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dropzoneTitle = document.getElementById('dropzoneTitle');
const dropzoneMeta = document.getElementById('dropzoneMeta');
const queryInput = document.getElementById('queryInput');
const graphType = document.getElementById('graphType');
const runBtn = document.getElementById('runBtn');
const statusLine = document.getElementById('statusLine');
const readout = document.getElementById('readout');
const statStrip = document.getElementById('statStrip');
const graphStage = document.getElementById('graphStage');
const insightText = document.getElementById('insightText');
const sqlText = document.getElementById('sqlText');
const dataTable = document.getElementById('dataTable');
const rowCount = document.getElementById('rowCount');

let selectedFile = null;
let statusTimer = null;

const STATUS_FRAMES = [
  'reading columns',
  'reasoning about the question',
  'writing sql',
  'querying the table',
  'building the chart',
  'drafting field notes',
];

function setStatus(text, kind) {
  statusLine.textContent = text || '';
  statusLine.className = 'status-line' + (kind ? ' ' + kind : '');
}

function startPulse() {
  let i = 0;
  setStatus(STATUS_FRAMES[0]);
  statusTimer = setInterval(() => {
    i = (i + 1) % STATUS_FRAMES.length;
    setStatus(STATUS_FRAMES[i]);
  }, 1400);
}

function stopPulse() {
  clearInterval(statusTimer);
  statusTimer = null;
}

// --- File selection ---

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    setStatus('Only .csv files are supported.', 'error');
    return;
  }
  selectedFile = file;
  dropzoneTitle.textContent = file.name;
  dropzoneMeta.textContent = `${(file.size / 1024).toFixed(1)} KB — click to replace`;
  updateRunState();
}

queryInput.addEventListener('input', updateRunState);

function updateRunState() {
  runBtn.disabled = !(selectedFile && queryInput.value.trim().length > 0);
}

// --- Run analysis ---

runBtn.addEventListener('click', async () => {
  if (!selectedFile || !queryInput.value.trim()) return;

  runBtn.disabled = true;
  runBtn.classList.add('running');
  setStatus('', null);
  startPulse();

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('query', queryInput.value.trim());
  formData.append('graph_type', graphType.value);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: formData });
    const payload = await res.json();

    if (!res.ok) {
      throw new Error(payload.detail || `Request failed (${res.status})`);
    }

    renderResult(payload);
    setStatus('done', null);
  } catch (err) {
    setStatus(err.message || 'Something went wrong.', 'error');
  } finally {
    stopPulse();
    runBtn.disabled = false;
    runBtn.classList.remove('running');
  }
});

// --- Rendering ---

function renderResult(payload) {
  readout.hidden = false;

  // Stats
  const rows = payload.data || [];
  const cols = rows.length ? Object.keys(rows[0]).length : 0;
  statStrip.innerHTML = '';
  addStat('ROWS', rows.length);
  addStat('COLUMNS', cols);
  addStat('CHART', payload.graph_type_used ? payload.graph_type_used.toUpperCase() : '—');

  // Graph
  graphStage.innerHTML = '';
  if (payload.graph) {
    const fig = payload.graph;
    const layout = Object.assign({}, fig.layout, {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: 'IBM Plex Mono, monospace', color: '#E7EDE9', size: 12 },
      margin: { t: 30, r: 20, l: 50, b: 50 },
      colorway: ['#E8B04B', '#4FD1C5', '#E8735A', '#8FA39C', '#B98CD9', '#7FB88A'],
    });
    Plotly.newPlot(graphStage, fig.data, layout, { responsive: true, displaylogo: false });
  } else {
    graphStage.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted);font-family:var(--mono);font-size:13px;">No chart could be built from this result — try a different question or chart type.</div>';
  }

  // Insight
  insightText.textContent = payload.insights || '—';

  // SQL
  sqlText.textContent = payload.query_used || '';

  // Table
  rowCount.textContent = rows.length;
  renderTable(rows);
}

function addStat(key, value) {
  const el = document.createElement('div');
  el.className = 'stat';
  el.innerHTML = `<span class="stat-value">${value}</span><span class="stat-key">${key}</span>`;
  statStrip.appendChild(el);
}

function renderTable(rows) {
  dataTable.innerHTML = '';
  if (!rows.length) return;

  const cols = Object.keys(rows[0]);
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  cols.forEach(c => {
    const th = document.createElement('th');
    th.textContent = c;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  dataTable.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.slice(0, 200).forEach(row => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      td.textContent = row[c];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  dataTable.appendChild(tbody);
}