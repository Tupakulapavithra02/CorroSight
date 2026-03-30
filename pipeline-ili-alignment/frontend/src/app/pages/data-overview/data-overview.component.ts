import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PlotlyModule } from 'angular-plotly.js';
import { MetricCardComponent } from '../../components/metric-card/metric-card.component';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-data-overview',
  standalone: true,
  imports: [CommonModule, PlotlyModule, MetricCardComponent],
  template: `
    <div class="page-container">
      <h1 class="page-title">Data Overview</h1>

      <!-- Run summary cards -->
      <div class="cards-row" *ngIf="runs.length">
        <app-metric-card *ngFor="let r of runs"
          [label]="'Run ' + r.year"
          [value]="r.anomaly_count"
          [subtitle]="r.girth_weld_count + ' girth welds | ' + r.vendor"
          [color]="runColors[r.year] || '#666'">
        </app-metric-card>
      </div>

      <!-- Completeness chart -->
      <div class="chart-card" *ngIf="completenessData.length">
        <h3>Column Completeness by Run (%)</h3>
        <plotly-plot [data]="completenessChart.data" [layout]="completenessChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Event type distribution -->
      <div class="chart-card" *ngIf="eventChart.data.length">
        <h3>Event Type Distribution</h3>
        <plotly-plot [data]="eventChart.data" [layout]="eventChart.layout" [config]="{responsive:true}"></plotly-plot>
      </div>

      <!-- Depth distributions -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="depthChart.data.length">
          <h3>Depth Distribution by Run</h3>
          <plotly-plot [data]="depthChart.data" [layout]="depthChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="clockChart.data.length">
          <h3>Clock Position Distribution (Pipe Cross-Section)</h3>
          <plotly-plot [data]="clockChart.data" [layout]="clockChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Length & Width -->
      <div class="chart-row">
        <div class="chart-card" *ngIf="lengthChart.data.length">
          <h3>Length Distribution by Run</h3>
          <plotly-plot [data]="lengthChart.data" [layout]="lengthChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
        <div class="chart-card" *ngIf="widthChart.data.length">
          <h3>Width Distribution by Run</h3>
          <plotly-plot [data]="widthChart.data" [layout]="widthChart.layout" [config]="{responsive:true}"></plotly-plot>
        </div>
      </div>

      <!-- Data quality table -->
      <div class="table-container" *ngIf="qualityReport.length">
        <h3 class="section-title">Data Quality Report</h3>
        <table class="full-width-table">
          <thead>
            <tr><th *ngFor="let col of qualityCols">{{col}}</th></tr>
          </thead>
          <tbody>
            <tr *ngFor="let row of qualityReport">
              <td *ngFor="let col of qualityCols">{{row[col]}}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    table { border-collapse: collapse; font-size: 13px; }
    th { background: #eceff1; padding: 8px 12px; text-align: left; font-weight: 500; }
    td { padding: 8px 12px; border-bottom: 1px solid #eceff1; }
  `]
})
export class DataOverviewComponent implements OnInit {
  runs: any[] = [];
  runColors: Record<number, string> = { 2007: '#636EFA', 2015: '#EF553B', 2022: '#00CC96' };
  completenessData: any[] = [];
  qualityReport: any[] = [];
  qualityCols: string[] = [];

  completenessChart: any = { data: [], layout: {} };
  eventChart: any = { data: [], layout: {} };
  depthChart: any = { data: [], layout: {} };
  lengthChart: any = { data: [], layout: {} };
  widthChart: any = { data: [], layout: {} };
  clockChart: any = { data: [], layout: {} };

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getSummary().subscribe(d => { this.runs = d.runs; });

    this.api.getQuality().subscribe(d => {
      this.qualityReport = d.quality_report;
      if (this.qualityReport.length) {
        this.qualityCols = Object.keys(this.qualityReport[0]);
      }
      this.completenessData = d.completeness;
      this.buildCompletenessChart();
    });

    this.api.getEda().subscribe(d => {
      this.buildEventChart(d.event_types);
      this.buildHistogram(d.depth_data, 'depthChart', 'Depth (% wall loss)', 'Number of Anomalies', 40);
      this.buildHistogram(d.length_data, 'lengthChart', 'Length (inches)', 'Number of Anomalies', 40);
      this.buildHistogram(d.width_data, 'widthChart', 'Width (inches)', 'Number of Anomalies', 40);
      this.buildClockChart(d.clock_data);
    });
  }

  buildCompletenessChart(): void {
    // Grouped bar chart: one bar group per column, colored by run year
    const years = [...new Set(this.completenessData.map((r: any) => r['Run Year']))].sort();
    const cols = [...new Set(this.completenessData.map((r: any) => r['Column']))];
    const colors: Record<string, string> = { '2007': '#636EFA', '2015': '#EF553B', '2022': '#00CC96' };

    // Friendly column display names
    const colLabels: Record<string, string> = {
      depth_pct: 'Depth %', length_in: 'Length', width_in: 'Width',
      clock_hours: 'Clock Pos', log_distance_ft: 'Log Dist',
      joint_number: 'Joint #', wall_thickness_in: 'Wall Thick',
      id_od: 'ID/OD', dist_to_us_weld_ft: 'Dist to Weld',
    };

    const traces: any[] = years.map(y => {
      const vals = cols.map(c => {
        const match = this.completenessData.find((r: any) => r['Run Year'] === y && r['Column'] === c);
        return match ? match['Completeness %'] : 0;
      });
      return {
        type: 'bar',
        name: `Run ${y}`,
        x: cols.map(c => colLabels[c] || c),
        y: vals,
        marker: { color: colors[String(y)] || '#999' },
        hovertemplate: `Run ${y}<br>%{x}: %{y:.1f}%<extra></extra>`,
      };
    });

    this.completenessChart = {
      data: traces,
      layout: {
        barmode: 'group',
        height: 320,
        margin: { t: 10, b: 80, l: 60, r: 20 },
        xaxis: { title: { text: 'Data Column' }, tickangle: -30 },
        yaxis: { title: { text: 'Completeness (%)' }, range: [0, 105] },
        legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      }
    };
  }

  buildEventChart(eventTypes: Record<string, Record<string, number>>): void {
    const allTypes = new Set<string>();
    Object.values(eventTypes).forEach(et => Object.keys(et).forEach(k => allTypes.add(k)));
    const types = [...allTypes].sort();
    const traces: any[] = [];
    const colors: Record<string, string> = { '2007': '#636EFA', '2015': '#EF553B', '2022': '#00CC96' };
    for (const [year, counts] of Object.entries(eventTypes)) {
      traces.push({
        type: 'bar', name: `Run ${year}`, x: types,
        y: types.map(t => (counts as any)[t] || 0),
        marker: { color: colors[year] },
      });
    }
    this.eventChart = {
      data: traces,
      layout: {
        barmode: 'group', height: 350, margin: { t: 10, b: 100, l: 60, r: 20 },
        xaxis: { title: { text: 'Event Type' }, tickangle: -45 },
        yaxis: { title: { text: 'Number of Features' } },
        legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      }
    };
  }

  buildHistogram(dataByYear: Record<string, number[]>, target: string, xLabel: string, yLabel: string, nbins: number): void {
    const colors: Record<string, string> = { '2007': '#636EFA', '2015': '#EF553B', '2022': '#00CC96' };
    const traces: any[] = [];
    for (const [year, values] of Object.entries(dataByYear)) {
      traces.push({
        type: 'histogram', x: values, name: `Run ${year}`, opacity: 0.6, nbinsx: nbins,
        marker: { color: colors[year] },
      });
    }
    (this as any)[target] = {
      data: traces,
      layout: {
        barmode: 'overlay', height: 300, margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: xLabel } },
        yaxis: { title: { text: yLabel } },
        legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      }
    };
  }

  buildClockChart(clockByYear: Record<string, number[]>): void {
    const colors: Record<string, string> = { '2007': '#636EFA', '2015': '#EF553B', '2022': '#00CC96' };
    const traces: any[] = [];
    for (const [year, values] of Object.entries(clockByYear)) {
      const theta = values.map(h => h * 30); // 12 hours = 360 degrees
      traces.push({
        type: 'barpolar', r: new Array(values.length).fill(1), theta, name: `Run ${year}`, opacity: 0.5,
        marker: { color: colors[year] },
      });
    }
    this.clockChart = {
      data: traces,
      layout: {
        polar: {
          angularaxis: { tickmode: 'array', tickvals: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                         ticktext: ['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], direction: 'clockwise' },
          radialaxis: { visible: false },
        },
        height: 350, margin: { t: 30, b: 30, l: 30, r: 30 }, barmode: 'overlay',
        legend: { orientation: 'h', y: -0.05, x: 0.5, xanchor: 'center' },
      }
    };
  }
}
