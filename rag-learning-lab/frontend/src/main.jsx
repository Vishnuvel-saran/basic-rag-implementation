import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowDown, Check, ChevronDown, CircleHelp, FileText, Filter, Gauge, GitBranch, LoaderCircle, MessageSquare, Network, Search, Send, UploadCloud, X } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const initialStages = ["Question", "Embedding", "Retrieval", "Top-K chunks", "Prompt", "LLM", "Answer"];

function App() {
  const [file, setFile] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [documentCap, setDocumentCap] = useState(4);
  const [chunkSize, setChunkSize] = useState(400);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [chunkingStrategy, setChunkingStrategy] = useState("fixed");
  const [semanticThreshold, setSemanticThreshold] = useState(0.75);
  const [chunkSearch, setChunkSearch] = useState("");
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [temperature, setTemperature] = useState(0);
  const [maxOutputTokens, setMaxOutputTokens] = useState(800);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const selectedDocument = file ? documents.find((document) => document.filename === file.name) : null;
  const atDocumentCap = documents.length >= documentCap;
  const canProcess = file && (!atDocumentCap || Boolean(selectedDocument));

  const loadDocuments = async () => {
    const response = await fetch(`${API_URL}/documents`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not load documents.");
    setDocuments(data.documents || []);
    setDocumentCap(data.max_documents ?? 4);
  };

  useEffect(() => {
    loadDocuments().catch((err) => setError(err.message));
  }, []);

  const visibleChunks = useMemo(() => {
    const query = chunkSearch.trim().toLowerCase();
    if (!query) return chunks;
    return chunks.filter((chunk) => `${chunk.chunk_id} ${chunk.context} ${chunk.metadata?.page_number || ""}`.toLowerCase().includes(query));
  }, [chunks, chunkSearch]);

  const chooseFile = (selectedFile) => {
    if (!selectedFile) return;
    if (selectedFile.type !== "application/pdf" && !selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    setFile(selectedFile);
    setError("");
  };

  const processDocument = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setStage("Embedding");
    try {
      const body = new FormData();
      body.append("file", file);
      const params = new URLSearchParams({
        chunking_strategy: chunkingStrategy,
        chunk_size: String(chunkSize),
      });
      if (["fixed", "paragraph", "recursive"].includes(chunkingStrategy)) {
        params.set("chunk_overlap", String(chunkOverlap));
      }
      if (chunkingStrategy === "agentic") {
        params.set("chunk_overlap", String(chunkOverlap));
      }
      if (chunkingStrategy === "semantic") {
        params.set("semantic_threshold", String(semanticThreshold));
      }
      const response = await fetch(`${API_URL}/documents/upload?${params}`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Document processing failed.");
      setChunks(data.chunks || []);
      setResult(null);
      await loadDocuments();
      setStage("Answer");
    } catch (err) {
      setError(err.message);
      setStage("");
    } finally {
      setBusy(false);
    }
  };

  const removeDocument = async (documentId) => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not remove document.");
      if (file?.name === documentId) {
        setFile(null);
        setChunks([]);
      }
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;
    setBusy(true);
    setError("");
    setStage("Question");
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), top_k: topK, temperature, max_output_tokens: maxOutputTokens, model }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Question failed.");
      setStage("Answer");
      setResult(data);
      setHistory((items) => [{ question: data.question, latency: data.telemetry?.latency_ms, input: data.telemetry?.input_tokens, output: data.telemetry?.output_tokens }, ...items].slice(0, 6));
    } catch (err) {
      setError(err.message);
      setStage("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><Network size={17} /></span><span>RAG / learning lab</span></div>
        <div className="status"><span className="status-dot" /> local workspace <span className="divider" /> OpenRouter</div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">RETRIEVAL OBSERVATORY <span>01</span></p>
          <h1>See the answer<br /><em>take shape.</em></h1>
          <p className="hero-copy">A small, inspectable RAG workbench. Upload a document, tune the split, then follow every retrieved fragment into the final answer.</p>
        </div>
        <div className="hero-note"><GitBranch size={18} /><span>Every request leaves a trail.</span></div>
      </section>

      <section className="workspace-grid">
        <aside className="left-rail">
          <PanelLabel number="01" title="Document input" icon={<FileText size={16} />} />
          <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}>
            <UploadCloud size={28} strokeWidth={1.5} />
            <strong>Drag & drop PDF</strong>
            <span>or choose a file from your machine</span>
            <button className="button ghost" onClick={() => fileInputRef.current?.click()}>Browse PDF</button>
            <input ref={fileInputRef} type="file" accept="application/pdf,.pdf" hidden onChange={(event) => chooseFile(event.target.files[0])} />
          </div>
          {file && <div className="file-chip"><FileText size={15} /><span>{file.name}</span><button onClick={() => setFile(null)} aria-label="Remove file"><X size={14} /></button></div>}
          <div className="control-block">
            <div className="control-heading"><span>Chunking controls</span><CircleHelp size={14} /></div>
            <label className="strategy-control"><span>Chunking strategy</span><select value={chunkingStrategy} onChange={(event) => setChunkingStrategy(event.target.value)}><option value="fixed">Fixed-size</option><option value="sentence">Sentence-based</option><option value="paragraph">Paragraph-based</option><option value="recursive">Recursive</option><option value="semantic">Semantic</option><option value="agentic">Agentic (LLM-guided)</option></select></label>
            <NumberControl label="Chunk size" value={chunkSize} onChange={setChunkSize} suffix="words" />
            {(["fixed", "recursive", "agentic"].includes(chunkingStrategy)) && <NumberControl label="Overlap" value={chunkOverlap} onChange={setChunkOverlap} suffix="words" />}
            {chunkingStrategy === "semantic" && <NumberControl label="Threshold" value={semanticThreshold} onChange={setSemanticThreshold} suffix="0–1" step="0.05" min="0" max="1" />}
            <button className="button primary full" disabled={!canProcess || busy} onClick={processDocument}>{busy && stage === "Embedding" ? <LoaderCircle className="spin" size={15} /> : <ArrowDown size={15} />} {busy && stage === "Embedding" ? "Processing..." : selectedDocument ? "Replace document" : "Process document"}</button>
          </div>
          <div className="document-manager">
            <div className="document-manager-heading"><span>Indexed documents</span><b>{documents.length} / {documentCap}</b></div>
            {documents.length ? documents.map((document) => <div className="document-row" key={document.document_id}><FileText size={15} /><div><strong>{document.filename}</strong><small>{document.page_count} pages · {document.chunk_count} chunks</small></div><button onClick={() => removeDocument(document.document_id)} disabled={busy} aria-label={`Remove ${document.filename}`}><X size={14} /></button></div>) : <span className="document-empty">No documents indexed yet.</span>}
          </div>
          {atDocumentCap && !selectedDocument && <div className="cap-note">Document limit reached. Choose an existing filename to replace it, or remove a document.</div>}
        </aside>

        <section className="main-column">
          <PanelLabel number="02" title="Ask the document" icon={<MessageSquare size={16} />} />
          <div className="ask-box">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") askQuestion(); }} placeholder="Ask a question about your document..." rows="3" />
            <div className="ask-footer"><span className="hint">Ctrl / Cmd + Enter to ask</span><div className="ask-actions"><label>Top-K <input className="top-k-input" type="number" min="1" step="1" value={topK} onChange={(event) => setTopK(Math.max(1, Number(event.target.value) || 1))} /></label><button className="button accent" disabled={!question.trim() || busy} onClick={askQuestion}>{busy && stage !== "Embedding" ? <LoaderCircle className="spin" size={15} /> : <Send size={15} />} Ask</button></div></div>
          </div>
          {error && <div className="error-banner">{error}</div>}
          <Pipeline active={stage} />
          <div className="llm-controls"><span>LLM experiment controls</span><label>Model<input value={model} onChange={(event) => setModel(event.target.value)} /></label><label>Temperature<input type="number" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label><label>Max tokens<input type="number" min="1" value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(Number(event.target.value))} /></label></div>
          <ResultArea result={result} />
        </section>
      </section>

      <section className="chunks-section">
        <div className="section-heading"><PanelLabel number="03" title="Document chunks" icon={<FileText size={16} />} /><span className="count-pill">{visibleChunks.length} / {chunks.length}</span></div>
        <div className="chunk-toolbar"><div className="search-field"><Search size={15} /><input value={chunkSearch} onChange={(event) => setChunkSearch(event.target.value)} placeholder="Filter chunks by text, page, or ID..." /></div><span className="muted">{chunks.length} chunks in latest upload · size {chunkSize}</span></div>
        <div className="chunks-grid">{visibleChunks.length ? visibleChunks.map((chunk, index) => <ChunkCard key={`${chunk.chunk_id}-${index}`} chunk={chunk} index={index} />) : <EmptyChunks hasDocument={chunks.length > 0} />}</div>
      </section>

      <section className="telemetry-section">
        <PanelLabel number="04" title="Request history" icon={<Activity size={16} />} />
        <div className="history-table"><div className="history-row history-head"><span>Question</span><span>Latency</span><span>Input</span><span>Output</span></div>{history.length ? history.map((item, index) => <div className="history-row" key={`${item.question}-${index}`}><span>{item.question}</span><span>{item.latency ? `${item.latency} ms` : "—"}</span><span>{item.input ?? "—"}</span><span>{item.output ?? "—"}</span></div>) : <div className="history-empty">Ask a question to start a trace.</div>}</div>
      </section>
      <footer><span>RAG LEARNING LAB</span><span>Backend observability · v0.1</span></footer>
    </main>
  );
}

function PanelLabel({ number, title, icon }) { return <div className="panel-label"><span className="label-number">{number}</span><span className="label-icon">{icon}</span><h2>{title}</h2></div>; }
function NumberControl({ label, value, onChange, suffix, step = "1", min = "1", max }) { return <label className="number-control"><span>{label}</span><div><input type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /><small>{suffix}</small></div></label>; }
function Pipeline({ active }) { return <div className="pipeline">{initialStages.map((item, index) => <div className={`pipeline-step ${active === item || (active === "Answer" && item === "Answer") ? "active" : ""}`} key={item}><span className="pipeline-node">{active === item ? <LoaderCircle className="spin" size={14} /> : active === "Answer" && item === "Answer" ? <Check size={14} /> : index + 1}</span><span>{item}</span>{index < initialStages.length - 1 && <i />}</div>)}</div>; }
function ChunkCard({ chunk, index }) { const pages = chunk.metadata?.page_ids || [chunk.metadata?.page_number ?? "—"]; return <article className="chunk-card"><div className="chunk-card-top"><span className="chunk-id">{chunk.chunk_id || `chunk_${String(index + 1).padStart(3, "0")}`}</span><span>pages {pages.join(", ")}</span></div><p>{chunk.context}</p><div className="chunk-meta"><span>{chunk.context?.length || 0} chars</span><span>~{Math.ceil((chunk.context || "").split(/\s+/).filter(Boolean).length)} tokens</span></div></article>; }
function EmptyChunks({ hasDocument }) { return <div className="empty-state"><FileText size={23} /><strong>{hasDocument ? "No chunks match your filter" : "Your chunks will appear here"}</strong><span>{hasDocument ? "Try a different search term." : "Upload and process a PDF to inspect the split."}</span></div>; }
function ResultArea({ result }) { if (!result) return <div className="result-placeholder"><Gauge size={23} /><span>Your answer will appear first, followed by the evidence trail.</span></div>; return <div className="result-area"><div className="answer-card"><div className="answer-kicker"><span className="live-dot" /> FINAL ANSWER</div><p>{result.answer}</p><div className="sources">{result.sources?.map((source) => <span key={source}>{source}</span>)}</div></div><Details title="Retrieved chunks" icon={<Search size={15} />} open><div className="retrieved-list">{result.retrieved_chunks?.map((chunk) => { const pages = chunk.metadata?.page_ids || [chunk.metadata?.page_number ?? "—"]; return <div className="retrieved-item" key={chunk.chunk_id}><div className="retrieved-top"><b>{chunk.chunk_id}</b><span className="score">{chunk.similarity_score != null ? chunk.similarity_score.toFixed(3) : "—"} similarity</span></div><small>pages {pages.join(", ")}</small><p>{chunk.context}</p></div>; })}</div></Details><Details title="Prompt sent to LLM" icon={<MessageSquare size={15} />}><pre className="prompt-view">{result.prompt}</pre></Details><div className="detail-grid"><Details title="Retrieval details" icon={<Network size={15} />} open><MetricGrid items={[["Embedding model", result.retrieval?.embedding_model], ["Dimension", result.retrieval?.embedding_dimension || "—"], ["Metric", result.retrieval?.similarity_metric], ["Searched", result.retrieval?.total_chunks_searched], ["Top-K", result.retrieval?.top_k]]} /></Details><Details title="LLM configuration" icon={<Gauge size={15} />}><MetricGrid items={[["Model", result.llm?.model], ["Temperature", result.llm?.temperature], ["Max output", result.llm?.max_output_tokens]]} /></Details><Details title="Telemetry" icon={<Activity size={15} />}><MetricGrid items={[["Input tokens", result.telemetry?.input_tokens ?? "Not provided"], ["Output tokens", result.telemetry?.output_tokens ?? "Not provided"], ["Total tokens", result.telemetry?.total_tokens ?? "Not provided"], ["Latency", `${result.telemetry?.latency_ms ?? "—"} ms`], ["Cost", result.telemetry?.estimated_cost ?? "Not provided"]]} /><span className="telemetry-note">Token values are {result.telemetry?.token_source === "actual" ? "reported by the provider." : "not supplied by the provider."}</span></Details></div></div>; }
function Details({ title, icon, children, open = false }) { return <details open={open} className="details"><summary>{icon}<span>{title}</span><ChevronDown size={15} /></summary><div className="details-body">{children}</div></details>; }
function MetricGrid({ items }) { return <div className="metric-grid">{items.map(([label, value]) => <div key={label}><span>{label}</span><b>{value ?? "—"}</b></div>)}</div>; }

export default App;

createRoot(document.getElementById("root")).render(<App />);
