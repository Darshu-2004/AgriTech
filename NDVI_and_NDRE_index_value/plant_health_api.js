/**
 * plant_health_api.js
 * =====================================================================
 * Frontend API helper for the Plant NDVI/NDRE Health Pipeline
 * Generated: 2026-06-12
 *
 * Usage (ESM / bundler):
 *   import PlantHealthAPI from './plant_health_api.js';
 *   const api = new PlantHealthAPI();
 *   await api.init();
 *
 * Usage (plain HTML):
 *   <script src="plant_health_api.js"></script>
 *   <script>
 *     const api = new PlantHealthAPI();
 *     api.init().then(() => { ... });
 *   </script>
 * =====================================================================
 */

class PlantHealthAPI {
  /**
   * @param {string} [dataDir='./data'] - Path to the data/ folder
   */
  constructor(dataDir = './data') {
    this.dataDir = dataDir;
    this._plants   = null;  // Array of all plant objects
    this._summary  = null;  // KPI summary + chart data
    this._insights = null;  // Full ML insights report
    this._sectors  = null;  // Sector-level stats array
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  /**
   * Load all data files. Must be called before any other method.
   * @returns {Promise<PlantHealthAPI>} resolves to `this` for chaining
   */
  async init() {
    const [plants, summary, insights, sectors] = await Promise.all([
      this._fetch('plants.json'),
      this._fetch('summary.json'),
      this._fetch('insights.json'),
      this._fetch('sectors.json'),
    ]);
    this._plants   = plants;
    this._summary  = summary;
    this._insights = insights;
    this._sectors  = sectors;
    console.log(`[PlantHealthAPI] Loaded ${this._plants.length.toLocaleString()} plants across ${this._sectors.length} sectors.`);
    return this;
  }

  async _fetch(filename) {
    const res = await fetch(`${this.dataDir}/${filename}`);
    if (!res.ok) throw new Error(`Failed to load ${filename}: ${res.status}`);
    return res.json();
  }

  // ── Plants ────────────────────────────────────────────────────────────────

  /** Return all plants */
  getAllPlants() {
    return this._plants;
  }

  /**
   * Get a single plant by its plant_id
   * @param {string} plantId e.g. "PLT-S00-R001-C001-8F00DB"
   */
  getPlantById(plantId) {
    return this._plants.find(p => p.plant_id === plantId) || null;
  }

  /**
   * Get plants by sector label
   * @param {string} sector e.g. "S00"
   */
  getBySector(sector) {
    return this._plants.filter(p => p.sector === sector);
  }

  /**
   * Get plants by row and column index (grid position)
   * @param {number} row
   * @param {number} col
   */
  getByPosition(row, col) {
    return this._plants.filter(p => p.row_index === row && p.col_index === col);
  }

  /**
   * Get plants by growth stage
   * @param {'Seedling'|'Vegetative'|'Flowering'|'Fruiting'|'Mature'} stage
   */
  getByGrowthStage(stage) {
    return this._plants.filter(p => p.growth_stage === stage);
  }

  /** Get all nitrogen-deficient plants */
  getDeficientPlants() {
    return this._plants.filter(p => p.is_nitrogen_deficient);
  }

  /** Get all plants with predicted yield < 20 kg */
  getLowYieldPlants() {
    return this._plants.filter(p => p.is_low_yield);
  }

  /** Get high-priority plants: nitrogen-deficient AND low yield */
  getHighPriorityPlants() {
    return this._plants.filter(p => p.is_high_priority);
  }

  /**
   * Advanced filter — pass any predicate function
   * @param {Function} predicateFn e.g. p => p.ndvi > 0.5 && p.sector === 'S00'
   */
  filterPlants(predicateFn) {
    return this._plants.filter(predicateFn);
  }

  /**
   * Sort plants by a field
   * @param {string} field e.g. 'yield_kg', 'ndvi', 'ndre'
   * @param {'asc'|'desc'} [order='desc']
   */
  sortBy(field, order = 'desc') {
    return [...this._plants].sort((a, b) =>
      order === 'desc' ? (b[field] ?? 0) - (a[field] ?? 0) : (a[field] ?? 0) - (b[field] ?? 0)
    );
  }

  // ── Summary / KPIs ────────────────────────────────────────────────────────

  /**
   * Get high-level KPI summary.
   * @returns {{
   *   total_plants, avg_ndvi, avg_ndre, total_yield_kg, avg_yield_kg,
   *   n_def_pct, n_def_count, healthy_pct, moderate_pct, stressed_pct, critical_pct,
   *   healthy, moderate, stressed, critical,
   *   ndvi_max, ndvi_min, ndre_max, ndre_min
   * }}
   */
  getSummary() {
    return this._summary?.summary ?? {};
  }

  /**
   * Get chart-ready data for histograms, donut charts, sector bars, etc.
   * @returns {{
   *   ndvi_bins, ndvi_hist,
   *   ndre_bins, ndre_hist,
   *   stage_names, stage_counts,
   *   sector_stats,
   *   health, health_labels
   * }}
   */
  getChartData() {
    return this._summary?.chart_data ?? {};
  }

  // ── Insights (ML Report) ──────────────────────────────────────────────────

  /** Full ML insights report */
  getInsights() {
    return this._insights ?? {};
  }

  /** Yield analysis sub-object */
  getYieldAnalysis() {
    return this._insights?.yield_analysis ?? {};
  }

  /** Nitrogen analysis sub-object */
  getNitrogenAnalysis() {
    return this._insights?.nitrogen_analysis ?? {};
  }

  /** Growth stage distribution { Mature: N, Seedling: N, ... } */
  getGrowthStageDistribution() {
    return this._insights?.growth_stage_distribution ?? {};
  }

  /** Prioritised recommendations array */
  getRecommendations() {
    return this._insights?.recommendations ?? [];
  }

  /** Per-sector performance { S00: { plant_count, avg_yield_kg, ... }, ... } */
  getSectorPerformance() {
    return this._insights?.sector_performance ?? {};
  }

  // ── Sectors ───────────────────────────────────────────────────────────────

  /** Full sector stats array (14 sectors S00–S13) */
  getAllSectors() {
    return this._sectors ?? [];
  }

  /**
   * Get stats for one sector
   * @param {string} sectorName e.g. "S02"
   */
  getSectorStats(sectorName) {
    return this._sectors?.find(s => s.name === sectorName) ?? null;
  }

  /** Sector with the highest average NDVI */
  getBestSector() {
    return [...(this._sectors ?? [])].sort((a, b) => b.ndvi_mean - a.ndvi_mean)[0] ?? null;
  }

  /** Sector with the lowest average NDVI (most stressed) */
  getWorstSector() {
    return [...(this._sectors ?? [])].sort((a, b) => a.ndvi_mean - b.ndvi_mean)[0] ?? null;
  }

  // ── Color Helpers ─────────────────────────────────────────────────────────

  /**
   * Returns a hex color for a given NDVI value
   * (used for map marker coloring)
   */
  static ndviColor(ndvi) {
    if (ndvi === null || ndvi === undefined) return '#6b7280';
    if (ndvi < 0)    return '#ef4444';  // Critical / water
    if (ndvi < 0.2)  return '#f97316';  // Bare / stressed
    if (ndvi < 0.4)  return '#eab308';  // Sparse
    if (ndvi < 0.6)  return '#84cc16';  // Moderate
    if (ndvi < 0.8)  return '#22c55e';  // Good
    return '#16a34a';                   // Very dense / healthy
  }

  /**
   * Returns a hex color for a given NDRE value
   */
  static ndreColor(ndre) {
    if (ndre === null || ndre === undefined) return '#6b7280';
    if (ndre < 0)    return '#ef4444';
    if (ndre < 0.1)  return '#f97316';
    if (ndre < 0.25) return '#eab308';
    if (ndre < 0.4)  return '#84cc16';
    if (ndre < 0.55) return '#22c55e';
    return '#16a34a';
  }

  /**
   * Returns a hex color for a growth stage name
   */
  static growthStageColor(stage) {
    const map = {
      Seedling:   '#8b5cf6',
      Vegetative: '#3b82f6',
      Flowering:  '#f59e0b',
      Fruiting:   '#10b981',
      Mature:     '#22c55e',
    };
    return map[stage] ?? '#6b7280';
  }

  /**
   * Returns a hex color for a predicted yield value
   */
  static yieldColor(yieldKg) {
    if (yieldKg === null || yieldKg === undefined) return '#6b7280';
    if (yieldKg >= 60) return '#16a34a';
    if (yieldKg >= 40) return '#22c55e';
    if (yieldKg >= 20) return '#f59e0b';
    return '#ef4444';
  }

  /**
   * Returns a hex color for a nitrogen status string
   */
  static nitrogenColor(status) {
    return status === 'Deficient' ? '#ef4444' : '#22c55e';
  }
}

// ── Export (supports both ESM and plain <script> usage) ───────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PlantHealthAPI;
} else if (typeof window !== 'undefined') {
  window.PlantHealthAPI = PlantHealthAPI;
}
