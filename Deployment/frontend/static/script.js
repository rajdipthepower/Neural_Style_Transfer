// --- Global Application State ---
let contentFile = null;             // Selected content Image File
let contentBlobUrl = null;          // Object URL for content preview
let selectedStyleId = null;         // Selected preset ID (e.g. 'starry_night')
let customStyleFile = null;         // Uploaded custom style File
let customStyleBlobUrl = null;      // Object URL for custom style preview
let currentScale = 'medium';        // 'fine' | 'medium' | 'broad'
let currentActiveTab = 'portrait';  // Current visible tab category
let allStylesData = [];             // Styles retrieved from GET /api/styles
let stylizedBlobUrl = null;         // Object URL of the stylization result

// --- DOM Element References ---
const errorBanner = document.getElementById('error-banner');
const errorMessage = document.getElementById('error-message');
const errorCloseBtn = document.getElementById('error-close-btn');
let errorTimeoutId = null;

// Content Upload Elements
const contentDropzone = document.getElementById('content-dropzone');
const contentInput = document.getElementById('content-input');
const contentPlaceholder = document.getElementById('content-placeholder');
const contentPreviewContainer = document.getElementById('content-preview-container');
const contentPreview = document.getElementById('content-preview');
const removeContentBtn = document.getElementById('remove-content-btn');

// Style Preset & Custom Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const presetGrid = document.getElementById('preset-grid');
const customStyleContainer = document.getElementById('custom-style-container');
const styleDropzone = document.getElementById('style-dropzone');
const styleInput = document.getElementById('style-input');
const stylePlaceholder = document.getElementById('style-placeholder');
const stylePreviewContainer = document.getElementById('style-preview-container');
const stylePreview = document.getElementById('style-preview');
const removeStyleBtn = document.getElementById('remove-style-btn');

// Control Elements
const alphaSlider = document.getElementById('alpha-slider');
const alphaVal = document.getElementById('alpha-val');
const scaleButtons = document.querySelectorAll('.scale-btn');
const stylizeBtn = document.getElementById('stylize-btn');

// Result Elements
const resultEmpty = document.getElementById('result-empty');
const resultSpinner = document.getElementById('result-spinner');
const resultDisplay = document.getElementById('result-display');
const resultImage = document.getElementById('result-image');
const downloadBtn = document.getElementById('download-btn');
const compareBtn = document.getElementById('compare-btn');

// Allowed MIME formats for client-side checks
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.jfif', '.png', '.webp'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

// ============================================================================
// Initialization
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
  setupErrorBanner();
  setupContentUpload();
  setupCustomStyleUpload();
  setupTabs();
  setupControls();
  setupResultInteractions();
  fetchStyles();
});

// ============================================================================
// Error Banner Helpers
// ============================================================================

function showError(msg) {
  clearTimeout(errorTimeoutId);
  errorMessage.textContent = msg;
  errorBanner.classList.remove('hidden');
  errorTimeoutId = setTimeout(hideError, 6000);
}

function hideError() {
  clearTimeout(errorTimeoutId);
  errorTimeoutId = null;
  errorBanner.classList.add('hidden');
  errorMessage.textContent = '';
}

function setupErrorBanner() {
  errorCloseBtn.addEventListener('click', hideError);
}

// ============================================================================
// Client-Side File Validation
// ============================================================================

function validateImageFile(file) {
  if (!file) return 'No file chosen.';
  const hasAllowedExtension = ALLOWED_EXTENSIONS.some(extension =>
    file.name.toLowerCase().endsWith(extension)
  );
  if (!ALLOWED_TYPES.includes(file.type) && !(file.type === '' && hasAllowedExtension)) {
    return 'Invalid file format. Please upload a JPEG, PNG, or WEBP image.';
  }
  if (file.size > MAX_FILE_SIZE) {
    return 'File is too large. Maximum file size allowed is 10 MB.';
  }
  return null;
}

function canDecodeImage(file) {
  return new Promise(resolve => {
    const imageUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      URL.revokeObjectURL(imageUrl);
      resolve(true);
    };
    image.onerror = () => {
      URL.revokeObjectURL(imageUrl);
      resolve(false);
    };
    image.src = imageUrl;
  });
}

// ============================================================================
// Content Image Upload Logic
// ============================================================================

function setupContentUpload() {
  // Click dropzone to open file picker
  contentDropzone.addEventListener('click', (e) => {
    if (e.target !== removeContentBtn) contentInput.click();
  });

  // Handle file picker selection
  contentInput.addEventListener('change', () => {
    if (contentInput.files.length > 0) {
      handleContentFile(contentInput.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(name => {
    contentDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      contentDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    contentDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      contentDropzone.classList.remove('dragover');
    });
  });

  contentDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleContentFile(e.dataTransfer.files[0]);
    }
  });

  // Remove button
  removeContentBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearContentFile();
  });
}

async function handleContentFile(file) {
  const error = validateImageFile(file);
  if (error) {
    showError(error);
    contentInput.value = '';
    return;
  }
  if (!await canDecodeImage(file)) {
    showError('This file is not a valid image.');
    contentInput.value = '';
    return;
  }
  hideError();

  contentFile = file;

  // Revoke previous blob URL to avoid memory leaks
  if (contentBlobUrl) URL.revokeObjectURL(contentBlobUrl);
  contentBlobUrl = URL.createObjectURL(file);

  contentPreview.src = contentBlobUrl;
  contentPlaceholder.classList.add('hidden');
  contentPreviewContainer.classList.remove('hidden');

  updateStylizeButtonState();
}

function clearContentFile() {
  contentFile = null;
  contentInput.value = '';
  if (contentBlobUrl) {
    URL.revokeObjectURL(contentBlobUrl);
    contentBlobUrl = null;
  }
  contentPreview.src = '';
  contentPreviewContainer.classList.add('hidden');
  contentPlaceholder.classList.remove('hidden');
  updateStylizeButtonState();
}

// ============================================================================
// Style Selection & Tabs Logic
// ============================================================================

async function fetchStyles() {
  try {
    const response = await fetch('/api/styles');
    if (!response.ok) throw new Error('Could not load preset styles.');
    allStylesData = await response.json();
    renderPresetGrid();
  } catch (err) {
    showError('Failed to load styles gallery from server.');
  }
}

function setupTabs() {
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      currentActiveTab = btn.getAttribute('data-tab');

      if (currentActiveTab === 'custom') {
        presetGrid.classList.add('hidden');
        customStyleContainer.classList.remove('hidden');
      } else {
        customStyleContainer.classList.add('hidden');
        presetGrid.classList.remove('hidden');
        renderPresetGrid();
      }
    });
  });
}

function renderPresetGrid() {
  presetGrid.innerHTML = '';

  // Filter styles by active category tab
  const filtered = allStylesData.filter(s => s.category.toLowerCase() === currentActiveTab);

  filtered.forEach(style => {
    const card = document.createElement('div');
    card.className = `preset-card ${selectedStyleId === style.id ? 'selected' : ''}`;
    card.setAttribute('data-id', style.id);

    card.innerHTML = `
      <img src="${style.thumb}" alt="${style.name}" loading="lazy">
      <span class="preset-name">${style.name}</span>
    `;

    card.addEventListener('click', () => {
      // Selecting a preset clears any custom uploaded style
      clearCustomStyleFile();
      selectedStyleId = style.id;

      // Update UI active card
      document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');

      updateStylizeButtonState();
    });

    presetGrid.appendChild(card);
  });
}

// ============================================================================
// Custom Style Upload Logic ("Upload Your Own")
// ============================================================================

function setupCustomStyleUpload() {
  styleDropzone.addEventListener('click', (e) => {
    if (e.target !== removeStyleBtn) styleInput.click();
  });

  styleInput.addEventListener('change', () => {
    if (styleInput.files.length > 0) {
      handleCustomStyleFile(styleInput.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    styleDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      styleDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    styleDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      styleDropzone.classList.remove('dragover');
    });
  });

  styleDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleCustomStyleFile(e.dataTransfer.files[0]);
    }
  });

  removeStyleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearCustomStyleFile();
  });
}

async function handleCustomStyleFile(file) {
  const error = validateImageFile(file);
  if (error) {
    showError(error);
    styleInput.value = '';
    return;
  }
  if (!await canDecodeImage(file)) {
    showError('This file is not a valid image.');
    styleInput.value = '';
    return;
  }
  hideError();

  // Selecting a custom upload clears any preset selection
  selectedStyleId = null;
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));

  customStyleFile = file;

  if (customStyleBlobUrl) URL.revokeObjectURL(customStyleBlobUrl);
  customStyleBlobUrl = URL.createObjectURL(file);

  stylePreview.src = customStyleBlobUrl;
  stylePlaceholder.classList.add('hidden');
  stylePreviewContainer.classList.remove('hidden');

  updateStylizeButtonState();
}

function clearCustomStyleFile() {
  customStyleFile = null;
  styleInput.value = '';
  if (customStyleBlobUrl) {
    URL.revokeObjectURL(customStyleBlobUrl);
    customStyleBlobUrl = null;
  }
  stylePreview.src = '';
  stylePreviewContainer.classList.add('hidden');
  stylePlaceholder.classList.remove('hidden');
  updateStylizeButtonState();
}

// ============================================================================
// Controls: Slider, Scale & Stylize Button
// ============================================================================

function setupControls() {
  // Alpha slider
  alphaSlider.addEventListener('input', () => {
    alphaVal.textContent = parseFloat(alphaSlider.value).toFixed(2);
  });

  // Scale buttons
  scaleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      scaleButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentScale = btn.getAttribute('data-scale');
    });
  });

  // Stylize execution
  stylizeBtn.addEventListener('click', runStylize);
}

function updateStylizeButtonState() {
  // Enabled only when content is provided AND exactly one style source is chosen
  const hasContent = contentFile !== null;
  const hasStyle = (selectedStyleId !== null) || (customStyleFile !== null);

  stylizeBtn.disabled = !(hasContent && hasStyle);
}

// ============================================================================
// Stylize API Execution
// ============================================================================

async function runStylize() {
  hideError();

  // Prepare Multipart Form Data
  const formData = new FormData();
  formData.append('content', contentFile);
  formData.append('alpha', alphaSlider.value);
  formData.append('scale', currentScale);

  // Send either style_id OR custom style file, never both
  if (selectedStyleId) {
    formData.append('style_id', selectedStyleId);
  } else if (customStyleFile) {
    formData.append('style', customStyleFile);
  } else {
    showError('Please choose a style before proceeding.');
    return;
  }

  // UI state: enter loading mode
  stylizeBtn.disabled = true;
  resultEmpty.classList.add('hidden');
  resultDisplay.classList.add('hidden');
  resultSpinner.classList.remove('hidden');

  try {
    const response = await fetch('/api/stylize', {
      method: 'POST',
      body: formData
    });

    // Handle non-OK HTTP responses
    if (!response.ok) {
      if (response.status === 503) {
        showError('Server is busy processing another image. Please try again in a few seconds.');
      } else {
        const errJson = await response.json().catch(() => ({}));
        showError(errJson.error || `Server responded with status ${response.status}`);
      }
      resultEmpty.classList.remove('hidden');
      return;
    }

    // Success: convert incoming JPEG bytes to an object URL
    const imageBlob = await response.blob();

    if (stylizedBlobUrl) URL.revokeObjectURL(stylizedBlobUrl);
    stylizedBlobUrl = URL.createObjectURL(imageBlob);

    resultImage.src = stylizedBlobUrl;
    downloadBtn.href = stylizedBlobUrl;

    resultDisplay.classList.remove('hidden');

  } catch (err) {
    showError('Network error: Unable to connect to the server.');
    resultEmpty.classList.remove('hidden');
  } finally {
    // Keep controls active and usable so the user can tweak parameters
    resultSpinner.classList.add('hidden');
    updateStylizeButtonState();
  }
}

// ============================================================================
// Result Interactions (Hold-To-Compare & Download)
// ============================================================================

function setupResultInteractions() {
  // Hold-to-compare: switch to original content photo on hold
  const showOriginal = (e) => {
    e.preventDefault();
    if (contentBlobUrl) resultImage.src = contentBlobUrl;
  };

  const showStylized = (e) => {
    e.preventDefault();
    if (stylizedBlobUrl) resultImage.src = stylizedBlobUrl;
  };

  // Mouse hold events
  compareBtn.addEventListener('mousedown', showOriginal);
  compareBtn.addEventListener('mouseup', showStylized);
  compareBtn.addEventListener('mouseleave', showStylized);

  // Touch hold events for mobile
  compareBtn.addEventListener('touchstart', showOriginal, { passive: false });
  compareBtn.addEventListener('touchend', showStylized, { passive: false });
  compareBtn.addEventListener('touchcancel', showStylized, { passive: false });
}