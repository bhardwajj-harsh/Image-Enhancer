document.addEventListener('DOMContentLoaded', () => {

  // File Upload handling
  const imageInput = document.getElementById('imageInput');
  const fileNameDisplay = document.getElementById('file-name');
  const imgOriginal = document.getElementById('imgOriginal');
  
  imageInput.addEventListener('change', (e) => {
    if(e.target.files.length > 0) {
      const file = e.target.files[0];
      fileNameDisplay.textContent = file.name;
      
      // Local preview
      const reader = new FileReader();
      reader.onload = (event) => {
        imgOriginal.src = event.target.result;
      }
      reader.readAsDataURL(file);
    } else {
      fileNameDisplay.textContent = 'Choose an Image';
    }
  });

  // Accordion Logic
  const accordions = document.querySelectorAll('.accordion-header');
  accordions.forEach(acc => {
    acc.addEventListener('click', () => {
      const parent = acc.parentElement;
      const content = parent.querySelector('.accordion-content');
      
      // Toggle current
      if(parent.classList.contains('active')){
        parent.classList.remove('active');
        content.style.display = 'none';
      } else {
        parent.classList.add('active');
        content.style.display = 'block';
      }
    });
  });

  // Dynamic parameters showing/hiding based on dropdowns
  const noiseType = document.getElementById('noiseType');
  const filterType = document.getElementById('filterType');
  const enhanceType = document.getElementById('enhanceType');

  function updateParamsDisplay(selectElem, prefix) {
    // Hide all
    const parent = selectElem.closest('.accordion-content');
    parent.querySelectorAll('.params-block').forEach(el => el.classList.add('hidden'));

    // Show selected
    const val = selectElem.value;
    if(val === 'Gaussian' && prefix === 'noise') document.getElementById('noiseParamsGaussian').classList.remove('hidden');
    if(val === 'Salt & Pepper' && prefix === 'noise') document.getElementById('noiseParamsSP').classList.remove('hidden');
    if(val === 'Speckle' && prefix === 'noise') document.getElementById('noiseParamsSpeckle').classList.remove('hidden');
    
    if(val === 'Gaussian Blur' && prefix === 'filter') document.getElementById('filterParamsGaussian').classList.remove('hidden');
    if(val === 'Median Filter' && prefix === 'filter') document.getElementById('filterParamsMedian').classList.remove('hidden');
    if(val === 'Bilateral Filter' && prefix === 'filter') document.getElementById('filterParamsBilateral').classList.remove('hidden');
    if(val === 'Non-local Means (NLM)' && prefix === 'filter') document.getElementById('filterParamsNLM').classList.remove('hidden');

    if(val === 'CLAHE' && prefix === 'enhance') document.getElementById('enhanceParamsCLAHE').classList.remove('hidden');
    if(val === 'Contrast Stretching' && prefix === 'enhance') document.getElementById('enhanceParamsStretching').classList.remove('hidden');
    if(val === 'Sharpening' && prefix === 'enhance') document.getElementById('enhanceParamsSharpening').classList.remove('hidden');
  }

  noiseType.addEventListener('change', () => updateParamsDisplay(noiseType, 'noise'));
  filterType.addEventListener('change', () => updateParamsDisplay(filterType, 'filter'));
  enhanceType.addEventListener('change', () => updateParamsDisplay(enhanceType, 'enhance'));

  // Range value labels: one listener + rAF so dragging sliders does not sync-layout thrash the page
  const dipForm = document.getElementById('dipForm');
  let rangeLabelRaf = 0;
  let pendingRangeInput = null;
  dipForm.addEventListener('input', (e) => {
    const t = e.target;
    if (t.type !== 'range') return;
    pendingRangeInput = t;
    if (rangeLabelRaf) return;
    rangeLabelRaf = requestAnimationFrame(() => {
      rangeLabelRaf = 0;
      const el = pendingRangeInput;
      if (!el || el.type !== 'range') return;
      const span = el.parentElement && el.parentElement.querySelector('label span');
      if (span) span.textContent = el.value;
    });
  });

  // Form Submission
  const loader = document.getElementById('loader');

  dipForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if(imageInput.files.length === 0){
      alert('Please upload an image first.');
      return;
    }

    loader.classList.add('active');
    loader.setAttribute('aria-hidden', 'false');
    
    const formData = new FormData(dipForm);
    const params = {};
    for (let [key, value] of formData.entries()) {
      // Try to parse numbers where applicable
      if(!isNaN(value) && value.trim() !== '') {
        params[key] = Number(value);
      } else {
        params[key] = value;
      }
    }

    const payload = new FormData();
    payload.append('file', imageInput.files[0]);
    payload.append('data', JSON.stringify(params));

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: payload
      });

      if(!response.ok) {
        throw new Error('Processing failed');
      }

      const result = await response.json();
      
      // Update DOM
      document.getElementById('imgOriginal').src = result.images.original;
      document.getElementById('imgNoisy').src = result.images.noisy;
      document.getElementById('imgDenoised').src = result.images.denoised;
      document.getElementById('imgEnhanced').src = result.images.enhanced;

      document.getElementById('mseNoisy').innerText = result.metrics.mse_noisy.toFixed(2);
      document.getElementById('psnrNoisy').innerText = result.metrics.psnr_noisy.toFixed(2);
      document.getElementById('mseDenoised').innerText = result.metrics.mse_denoised.toFixed(2);
      document.getElementById('psnrDenoised').innerText = result.metrics.psnr_denoised.toFixed(2);

    } catch (err) {
      console.error(err);
      alert('An error occurred during processing. Please try again.');
    } finally {
      loader.classList.remove('active');
      loader.setAttribute('aria-hidden', 'true');
    }
  });

});
