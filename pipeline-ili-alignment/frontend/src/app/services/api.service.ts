import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const BASE = 'http://localhost:8000/api';

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  getSummary(): Observable<any> {
    return this.http.get(`${BASE}/summary`);
  }

  getQuality(): Observable<any> {
    return this.http.get(`${BASE}/quality`);
  }

  getEda(): Observable<any> {
    return this.http.get(`${BASE}/eda`);
  }

  getAlignment(): Observable<any> {
    return this.http.get(`${BASE}/alignment`);
  }

  getMatches(pair: string): Observable<any> {
    return this.http.get(`${BASE}/matches/${pair}`);
  }

  getGrowth(pair: string): Observable<any> {
    return this.http.get(`${BASE}/growth/${pair}`);
  }

  getMultirun(): Observable<any> {
    return this.http.get(`${BASE}/multirun`);
  }

  getPipelineView(): Observable<any> {
    return this.http.get(`${BASE}/pipeline-view`);
  }

  exportExcel(): void {
    window.open(`${BASE}/export`, '_blank');
  }

  // ── AI-Powered Endpoints ──────────────────────────────────────────────────

  chat(message: string, history: { role: string; content: string }[]): Observable<any> {
    return this.http.post(`${BASE}/chat`, { message, history });
  }

  getAiReport(): Observable<any> {
    return this.http.get(`${BASE}/ai-report`);
  }

  getAiNarratives(): Observable<any> {
    return this.http.get(`${BASE}/ai-narratives`);
  }

  getAiInsights(chartType: string, data: any): Observable<any> {
    return this.http.post(`${BASE}/ai-insights`, { chart_type: chartType, data });
  }

  getNlChart(query: string): Observable<any> {
    return this.http.post(`${BASE}/nl-chart`, { query });
  }

  getVirtualIli(year: number): Observable<any> {
    return this.http.get(`${BASE}/virtual-ili/${year}`);
  }

  getIntegrityDashboard(): Observable<any> {
    return this.http.get(`${BASE}/integrity-dashboard`);
  }

  // ── API Key & Pipeline ────────────────────────────────────────────────────

  setApiKey(key: string): Observable<any> {
    return this.http.post(`${BASE}/set-api-key`, { key });
  }

  getAiStatus(): Observable<any> {
    return this.http.get(`${BASE}/ai-status`);
  }

  runPipeline(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${BASE}/run-pipeline`, formData);
  }

  getPipelineStatus(): Observable<any> {
    return this.http.get(`${BASE}/run-pipeline/status`);
  }
}
