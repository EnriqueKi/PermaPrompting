/**
 * Frontend JavaScript for Chromatography Analysis Tool
 * Handles file upload, API communication, and results display
 */

class ChromatographyAnalyzer {
  constructor() {
    this.selectedFile = null;
    this.API_BASE_URL = "http://localhost:8080";
    this.initializeEventListeners();
    this.checkAPIHealth();
  }

  initializeEventListeners() {
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");

    // Drag and drop functionality
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, this.preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(
        eventName,
        () => uploadArea.classList.add("dragover"),
        false
      );
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(
        eventName,
        () => uploadArea.classList.remove("dragover"),
        false
      );
    });

    uploadArea.addEventListener("drop", (e) => this.handleDrop(e), false);
    fileInput.addEventListener("change", (e) => this.handleFileSelect(e));
  }

  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  handleFile(file) {
    if (!file.type.startsWith("image/")) {
      this.showError(
        "Please select a valid image file (PNG, JPG, JPEG, BMP, TIFF)."
      );
      return;
    }

    // Check file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
      this.showError("File size too large. Maximum size is 16MB.");
      return;
    }

    this.selectedFile = file;
    this.showPreview(file);
    this.enableAnalyzeButton();
  }

  showPreview(file) {
    const imagePreview = document.getElementById("image-preview");
    const previewInfo = document.getElementById("preview-info");

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
      previewInfo.textContent = `File: ${file.name} (${this.formatFileSize(
        file.size
      )})`;
    };
    reader.readAsDataURL(file);
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  enableAnalyzeButton() {
    const analyzeButton = document.getElementById("analyze-button");
    analyzeButton.disabled = false;
  }

  async analyzeImage() {
    if (!this.selectedFile) {
      this.showError("Please select an image file first.");
      return;
    }

    const params = this.getAnalysisParameters();
    if (!this.validateParameters(params)) {
      return;
    }

    this.hideError();
    this.hideResults();
    this.showLoading();

    try {
      const result = await this.makeAPIRequest(params);
      this.displayResults(result);
    } catch (error) {
      console.error("Analysis error:", error);
      this.showError(error.message || "Analysis failed. Please try again.");
    } finally {
      this.hideLoading();
    }
  }

  getAnalysisParameters() {
    return {
      nRegions: parseInt(document.getElementById("n-regions").value),
      segmentationMethod: document.getElementById("segmentation-method").value,
      paperDiameter: parseFloat(
        document.getElementById("paper-diameter").value
      ),
    };
  }

  validateParameters(params) {
    if (params.nRegions < 1 || params.nRegions > 10) {
      this.showError("Number of regions must be between 1 and 10.");
      return false;
    }

    if (params.paperDiameter <= 0) {
      this.showError("Paper diameter must be greater than 0.");
      return false;
    }

    return true;
  }

  async makeAPIRequest(params) {
    const formData = new FormData();
    formData.append("file", this.selectedFile);
    formData.append("n_regions", params.nRegions.toString());
    formData.append("segmentation_method", params.segmentationMethod);
    formData.append("paper_diameter_cm", params.paperDiameter.toString());

    const response = await fetch(`${this.API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || result.error || "Analysis failed");
    }

    return result;
  }

  displayResults(results) {
    const resultsGrid = document.getElementById("results-grid");
    resultsGrid.innerHTML = "";

    // Create summary card
    this.createSummaryCard(results, resultsGrid);

    // Create zones card
    if (results.zones) {
      this.createZonesCard(results.zones, resultsGrid);
    }

    // Create radial analysis card
    if (results.radial_analysis) {
      this.createRadialAnalysisCard(results.radial_analysis, resultsGrid);
    }

    this.showResults();
  }

  createSummaryCard(results, container) {
    const summaryItems = [
      `Total Zones: ${results.summary?.total_zones || "N/A"}`,
      `Total Area: ${results.summary?.total_area_cm2?.toFixed(2) || "N/A"} cm²`,
      `Center: (${results.center?.[0]?.toFixed(1) || "N/A"}, ${
        results.center?.[1]?.toFixed(1) || "N/A"
      })`,
      `Method: ${results.analysis_parameters?.segmentation_method || "N/A"}`,
    ];

    const card = this.createResultCard("Analysis Summary", summaryItems);
    container.appendChild(card);
  }

  createZonesCard(zones, container) {
    const zoneItems = Object.entries(zones).map(([zoneKey, zoneData]) => {
      const color = zoneData.color;
      let colorStyle = this.getColorStyle(color);

      return {
        text: `${zoneData.zone_full_name || zoneKey}: ${
          zoneData.width_cm?.toFixed(3) || "N/A"
        } cm`,
        color: colorStyle,
      };
    });

    const card = this.createZoneCard("Zone Analysis", zoneItems);
    container.appendChild(card);
  }

  createRadialAnalysisCard(radialData, container) {
    const radialInfo = [];

    if (radialData.channel_development) {
      const ch = radialData.channel_development;
      radialInfo.push(`Channels: ${ch.total_channels || 0} total`);
      radialInfo.push(
        `Avg Length: ${ch.avg_channel_length_cm?.toFixed(2) || "N/A"} cm`
      );
    }

    if (radialData.spike_development) {
      const sp = radialData.spike_development;
      radialInfo.push(`Spikes: ${sp.total_spikes || 0} total`);
      radialInfo.push(
        `Density: ${sp.spike_density?.toFixed(2) || "N/A"}/10k pixels`
      );
    }

    if (radialInfo.length > 0) {
      const card = this.createResultCard("Radial Analysis", radialInfo);
      container.appendChild(card);
    }
  }

  getColorStyle(color) {
    if (color.red !== undefined) {
      return `rgb(${Math.round(color.red)}, ${Math.round(
        color.green
      )}, ${Math.round(color.blue)})`;
    } else if (color.gray !== undefined) {
      const gray = Math.round(color.gray);
      return `rgb(${gray}, ${gray}, ${gray})`;
    }
    return "#cccccc";
  }

  createResultCard(title, items) {
    const card = document.createElement("div");
    card.className = "result-card";

    const cardTitle = document.createElement("h3");
    cardTitle.textContent = title;
    card.appendChild(cardTitle);

    items.forEach((item) => {
      const itemDiv = document.createElement("div");
      itemDiv.textContent = item;
      itemDiv.style.marginBottom = "8px";
      card.appendChild(itemDiv);
    });

    return card;
  }

  createZoneCard(title, zoneItems) {
    const card = document.createElement("div");
    card.className = "result-card";

    const cardTitle = document.createElement("h3");
    cardTitle.textContent = title;
    card.appendChild(cardTitle);

    zoneItems.forEach((item) => {
      const zoneDiv = document.createElement("div");
      zoneDiv.className = "zone-item";

      const colorDiv = document.createElement("div");
      colorDiv.className = "zone-color";
      if (item.color) {
        colorDiv.style.backgroundColor = item.color;
      }

      const textDiv = document.createElement("div");
      textDiv.textContent = item.text;
      textDiv.style.flex = "1";

      zoneDiv.appendChild(colorDiv);
      zoneDiv.appendChild(textDiv);
      card.appendChild(zoneDiv);
    });

    return card;
  }

  showLoading() {
    document.getElementById("loading").classList.add("show");
  }

  hideLoading() {
    document.getElementById("loading").classList.remove("show");
  }

  showResults() {
    document.getElementById("results-section").classList.add("show");
  }

  hideResults() {
    document.getElementById("results-section").classList.remove("show");
  }

  showError(message) {
    const errorDiv = document.getElementById("error-message");
    errorDiv.textContent = message;
    errorDiv.classList.add("show");
  }

  hideError() {
    document.getElementById("error-message").classList.remove("show");
  }

  async checkAPIHealth() {
    try {
      const response = await fetch(`${this.API_BASE_URL}/health`);
      if (!response.ok) {
        this.showError(
          "Warning: Backend API is not responding. Please make sure the server is running on port 8080."
        );
      }
    } catch (error) {
      this.showError(
        "Warning: Cannot connect to backend API. Please make sure the server is running on port 8080."
      );
    }
  }
}

// Initialize the analyzer when the page loads
let analyzer;
document.addEventListener("DOMContentLoaded", () => {
  analyzer = new ChromatographyAnalyzer();
});

// Global function for the analyze button (called from HTML)
function analyzeImage() {
  if (analyzer) {
    analyzer.analyzeImage();
  }
}
