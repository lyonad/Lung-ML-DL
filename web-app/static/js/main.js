// ==================== Global State ====================
let currentStep = 1;
const totalSteps = 3;
const formData = {};

// ==================== DOM Elements ====================
const assessmentForm = document.getElementById('assessment-form');
const formSteps = document.querySelectorAll('.form-step');
const progressSteps = document.querySelectorAll('.progress-step');
const btnPrev = document.querySelector('.btn-prev');
const btnNext = document.querySelector('.btn-next');
const btnSubmit = document.querySelector('.btn-submit');
const assessmentSection = document.querySelector('.assessment-section');
const resultsSection = document.querySelector('.results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navLinks = document.querySelector('.nav-links');

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    setupSmoothScrolling();
    setupMobileMenu();
    setupScrollIndicator();
    setupScrollSpy();
    
    // Only initialize form navigation if form exists
    if (assessmentForm && btnPrev && btnNext && btnSubmit) {
        updateFormNavigation();
    }
});

// ==================== Event Listeners ====================
function initializeEventListeners() {
    // Form navigation - check if elements exist first
    if (btnPrev) {
        btnPrev.addEventListener('click', handlePrevStep);
    }
    if (btnNext) {
        btnNext.addEventListener('click', handleNextStep);
    }
    if (assessmentForm) {
        assessmentForm.addEventListener('submit', handleFormSubmit);
    }
    
    // New assessment button
    const btnNewAssessment = document.querySelector('.btn-new-assessment');
    if (btnNewAssessment) {
        btnNewAssessment.addEventListener('click', resetAssessment);
    }
    
    // Input validation
    const ageInput = document.getElementById('age');
    if (ageInput) {
        ageInput.addEventListener('input', validateAge);
    }
    
    // Radio/toggle selections - clear error state when selected
    const radioInputs = document.querySelectorAll('input[type="radio"]');
    radioInputs.forEach(input => {
        input.addEventListener('change', () => {
            // Clear error state when field is filled
            const formGroup = input.closest('.form-group');
            if (formGroup) {
                formGroup.classList.remove('field-error');
                formGroup.classList.add('has-value');
            }
        });
    });
    
    // Number inputs - clear error when user types
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', () => {
            const formGroup = input.closest('.form-group');
            if (formGroup && input.value) {
                formGroup.classList.remove('field-error');
            }
        });
    });
}

// ==================== Mobile Menu ====================
function setupMobileMenu() {
    if (mobileMenuToggle && navLinks) {
        mobileMenuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
        
        // Close menu when clicking a link
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
            });
        });
    }
}

// ==================== Scroll Indicator ====================
function setupScrollIndicator() {
    const scrollIndicator = document.querySelector('.scroll-indicator');
    if (scrollIndicator) {
        scrollIndicator.addEventListener('click', () => {
            const assessmentSection = document.getElementById('assessment');
            if (assessmentSection) {
                assessmentSection.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }
        });
        
        // Hide scroll indicator when user scrolls down
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            if (currentScroll > 100) {
                scrollIndicator.style.opacity = '0';
                scrollIndicator.style.pointerEvents = 'none';
            } else {
                scrollIndicator.style.opacity = '1';
                scrollIndicator.style.pointerEvents = 'auto';
            }
            lastScroll = currentScroll;
        });
    }
}

// ==================== Smooth Scrolling ====================
function setupSmoothScrolling() {
    // Handle all anchor links with hash (#)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offsetTop = target.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Handle navigation links for same-page scrolling
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', handleNavigation);
    });
}

function handleNavigation(e) {
    const href = this.getAttribute('href');
    const currentPath = window.location.pathname;
    
    // Check if it's a same-page navigation
    if (href === '/' && (currentPath === '/' || currentPath === '/index.html')) {
        // Scroll to top smoothly instead of reloading
        e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
        // Update active state
        updateActiveNavLink(this);
    } else if (href.startsWith('/') && !href.includes('#')) {
        // Different page - let it navigate normally but handle hash if present
        const targetPage = href.split('#')[0];
        const targetHash = href.split('#')[1];
        
        if (currentPath === targetPage || (currentPath === '/' && targetPage === '/index.html')) {
            // Same page, just scroll to section if hash exists
            if (targetHash) {
                e.preventDefault();
                const target = document.getElementById(targetHash);
                if (target) {
                    const offsetTop = target.offsetTop - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            }
        }
    }
    
    // Close mobile menu if open
    if (navLinks && navLinks.classList.contains('active')) {
        navLinks.classList.remove('active');
    }
}

function updateActiveNavLink(activeLink) {
    // Remove active class from all nav links
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
    });
    // Add active class to clicked link
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

// ==================== Scroll Spy for Active Nav Links ====================
function setupScrollSpy() {
    // Only run on homepage
    const currentPath = window.location.pathname;
    if (currentPath !== '/' && currentPath !== '/index.html') {
        return;
    }
    
    const sections = [
        { id: 'assessment', navHref: '/#assessment' },
        { id: 'how-it-works', navHref: '/#how-it-works' }
    ];
    
    let isScrolling = false;
    
    window.addEventListener('scroll', () => {
        if (isScrolling) return;
        
        const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        // If at the very top (first 100px), highlight Home
        if (scrollPosition < 100) {
            updateActiveNavByHref('/');
            return;
        }
        
        // If at the very bottom, highlight the last section
        if (scrollPosition + windowHeight >= documentHeight - 100) {
            updateActiveNavByHref('/#how-it-works');
            return;
        }
        
        // Check which section is in view
        for (let i = sections.length - 1; i >= 0; i--) {
            const section = document.getElementById(sections[i].id);
            if (section) {
                const sectionTop = section.offsetTop - 150;
                const sectionBottom = sectionTop + section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                    updateActiveNavByHref(sections[i].navHref);
                    return;
                }
            }
        }
    }, { passive: true });
}

function updateActiveNavByHref(href) {
    const navLinksElements = document.querySelectorAll('.nav-links a');
    navLinksElements.forEach(link => {
        const linkHref = link.getAttribute('href');
        if (linkHref === href || linkHref.endsWith(href)) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// ==================== Form Navigation ====================
function handlePrevStep() {
    if (currentStep > 1) {
        currentStep--;
        updateFormDisplay();
        updateFormNavigation();
    }
}

function handleNextStep() {
    if (validateCurrentStep()) {
        if (currentStep < totalSteps) {
            currentStep++;
            updateFormDisplay();
            updateFormNavigation();
        }
    } else {
        // Show error notification with helpful message
        let message = 'Please complete all required fields in this step';
        if (currentStep === 1) {
            message = 'Please enter your age and select your gender';
        } else if (currentStep === 2) {
            message = 'Please answer all lifestyle and risk factor questions';
        } else if (currentStep === 3) {
            message = 'Please answer all symptom questions';
        }
        showNotification(message, 'error');
        
        // Highlight missing fields
        highlightMissingFields();
    }
}

function updateFormDisplay() {
    // Update form steps
    formSteps.forEach((step, index) => {
        if (index + 1 === currentStep) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
    
    // Update progress indicators
    progressSteps.forEach((step, index) => {
        const stepNumber = index + 1;
        if (stepNumber < currentStep) {
            step.classList.add('completed');
            step.classList.remove('active');
        } else if (stepNumber === currentStep) {
            step.classList.add('active');
            step.classList.remove('completed');
        } else {
            step.classList.remove('active', 'completed');
        }
    });
    
    // Scroll to form
    const formTop = assessmentForm.offsetTop - 100;
    window.scrollTo({
        top: formTop,
        behavior: 'smooth'
    });
}

function updateFormNavigation() {
    if (!btnPrev || !btnNext || !btnSubmit) {
        return; // Form elements don't exist on this page
    }
    
    // Update previous button
    btnPrev.disabled = currentStep === 1;
    
    // Show/hide next and submit buttons
    if (currentStep === totalSteps) {
        btnNext.style.display = 'none';
        btnSubmit.style.display = 'inline-flex';
    } else {
        btnNext.style.display = 'inline-flex';
        btnSubmit.style.display = 'none';
    }
    
    // Enable next button - validation happens on click
    if (currentStep < totalSteps && btnNext) {
        btnNext.disabled = false;
    }
}

// ==================== Validation ====================
function validateAge(e) {
    const age = parseInt(e.target.value);
    if (age < 1) {
        e.target.value = 1;
    } else if (age > 120) {
        e.target.value = 120;
    }
}

function highlightMissingFields() {
    const currentStepElement = document.querySelector(`.form-step[data-step="${currentStep}"]`);
    if (!currentStepElement) return;
    
    // Remove previous highlights
    currentStepElement.querySelectorAll('.form-group').forEach(group => {
        group.classList.remove('field-error');
    });
    
    // Get ALL inputs to properly group radio buttons
    const allInputs = currentStepElement.querySelectorAll('input');
    const groupedInputs = {};
    
    allInputs.forEach(input => {
        const name = input.getAttribute('name');
        const type = input.getAttribute('type');
        const isRequired = input.hasAttribute('required');
        
        if (type === 'radio' && name) {
            if (!groupedInputs[name]) {
                groupedInputs[name] = { inputs: [], required: false };
            }
            groupedInputs[name].inputs.push(input);
            if (isRequired) {
                groupedInputs[name].required = true;
            }
        } else if ((type === 'number' || type === 'text') && isRequired) {
            if (!input.value || input.value.trim() === '') {
                input.closest('.form-group')?.classList.add('field-error');
            }
        }
    });
    
    // Check required radio groups
    for (const name in groupedInputs) {
        const group = groupedInputs[name];
        if (group.required) {
            const isChecked = group.inputs.some(input => input.checked);
            if (!isChecked) {
                group.inputs[0].closest('.form-group')?.classList.add('field-error');
            }
        }
    }
}

function validateCurrentStep() {
    const currentStepElement = document.querySelector(`.form-step[data-step="${currentStep}"]`);
    if (!currentStepElement) {
        console.error('Current step element not found');
        return false;
    }
    
    // Make sure we're checking the active step only
    if (!currentStepElement.classList.contains('active')) {
        console.warn('Step element is not active');
    }
    
    let allValid = true;
    const groupedInputs = {};
    const missingFields = [];
    
    // Get ALL inputs (not just required) to find radio groups
    const allInputs = currentStepElement.querySelectorAll('input');
    console.log(`Step ${currentStep}: Found ${allInputs.length} total inputs`);
    
    allInputs.forEach(input => {
        const name = input.getAttribute('name');
        const type = input.getAttribute('type');
        const isRequired = input.hasAttribute('required');
        
        // Handle radio buttons - group ALL radio buttons by name
        if (type === 'radio' && name) {
            if (!groupedInputs[name]) {
                groupedInputs[name] = { inputs: [], required: false };
            }
            groupedInputs[name].inputs.push(input);
            if (isRequired) {
                groupedInputs[name].required = true;
            }
        }
        // Handle text/number inputs that are required
        else if ((type === 'number' || type === 'text') && isRequired) {
            if (!input.value || input.value.trim() === '') {
                allValid = false;
                missingFields.push(name);
                console.log(`Missing ${type} input: ${name}`);
            }
        }
    });
    
    // Check if at least one radio in each REQUIRED group is selected
    const requiredGroups = Object.keys(groupedInputs).filter(name => groupedInputs[name].required);
    console.log(`Step ${currentStep}: Checking ${requiredGroups.length} required radio groups`);
    
    for (const name of requiredGroups) {
        const group = groupedInputs[name];
        const isChecked = group.inputs.some(input => input.checked);
        console.log(`Radio group "${name}": ${isChecked ? 'checked ✓' : 'NOT checked ✗'}`);
        if (!isChecked) {
            allValid = false;
            missingFields.push(name);
        }
    }
    
    // Debug logging
    if (!allValid) {
        console.log('❌ Validation failed. Missing fields:', missingFields);
    } else {
        console.log('✅ Validation passed for step', currentStep);
    }
    
    return allValid;
}

function collectFormData() {
    const data = {};
    const formElements = assessmentForm.elements;
    
    for (let element of formElements) {
        if (element.name && element.checked) {
            data[element.name] = element.value;
        } else if (element.name && element.type === 'number') {
            data[element.name] = parseInt(element.value);
        }
    }
    
    return data;
}

// ==================== Form Submission ====================
async function handleFormSubmit(e) {
    e.preventDefault();
    
    if (!validateCurrentStep()) {
        showNotification('Please complete all required fields', 'error');
        return;
    }
    
    const data = collectFormData();
    
    // Show loading
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            displayResults(result);
        } else {
            showNotification(result.error || 'An error occurred during prediction', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Failed to connect to the server. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

// ==================== Results Display ====================
function displayResults(data) {
    console.log('Displaying results:', data);
    
    // Hide assessment, show results
    assessmentSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Display overall risk
    displayOverallRisk(data.recommendation);
    
    // Display model predictions - check both possible key names
    const annData = data.predictions?.ann || data.predictions?.regularized_ann;
    const rfData = data.predictions?.rf || data.predictions?.random_forest;
    
    console.log('ANN Data:', annData);
    console.log('RF Data:', rfData);
    
    if (annData) {
        displayModelPrediction('ann', annData);
    }
    
    if (rfData) {
        displayModelPrediction('rf', rfData);
    }
    
    // Display input summary
    displayInputSummary(data.input_summary, data.timestamp);
}

function displayOverallRisk(recommendation) {
    const riskBadge = document.getElementById('overall-risk-badge');
    const riskText = document.getElementById('overall-risk-text');
    const confidenceValue = document.getElementById('confidence-value');
    const recommendationText = document.getElementById('recommendation-text');
    
    if (!recommendation) return;
    
    // Set risk level
    const riskLevel = recommendation.risk_level || 'Unknown';
    const confidence = recommendation.confidence || 0;
    const message = recommendation.message || '';
    const action = recommendation.action || '';
    
    riskText.textContent = riskLevel;
    confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
    recommendationText.textContent = `${message} ${action}`;
    
    // Add appropriate class
    riskBadge.className = 'risk-level-badge';
    if (riskLevel.toLowerCase().includes('low') || riskLevel.toLowerCase().includes('no')) {
        riskBadge.classList.add('low');
    } else {
        riskBadge.classList.add('high');
    }
}

function displayModelPrediction(modelType, prediction) {
    console.log(`Displaying ${modelType} prediction:`, prediction);
    
    // Parse probability (might be 0-1 range)
    const probability = parseFloat(prediction.probability) || 0;
    const percentage = (probability * 100).toFixed(1);
    
    // Get prediction class (could be 'prediction' or 'prediction_class')
    const predictionClass = prediction.prediction || prediction.prediction_class || 'Unknown';
    
    // Parse confidence (might be a string like "95.00%" or a number)
    let confidenceText = prediction.confidence;
    if (typeof confidenceText === 'string' && confidenceText.includes('%')) {
        // Already formatted
    } else if (typeof confidenceText === 'number') {
        confidenceText = `${(confidenceText * 100).toFixed(1)}%`;
    } else {
        confidenceText = percentage + '%';
    }
    
    // Update percentage text
    document.getElementById(`${modelType}-percentage`).textContent = `${percentage}%`;
    
    // Update prediction and confidence
    const predictionText = predictionClass.replace('_', ' ');
    document.getElementById(`${modelType}-prediction`).textContent = predictionText;
    document.getElementById(`${modelType}-confidence`).textContent = confidenceText;
    
    // Animate circle
    const circle = document.getElementById(`${modelType}-progress`);
    if (circle) {
        const circumference = 2 * Math.PI * 45; // radius = 45
        const offset = circumference - (percentage / 100) * circumference;
        
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);
        
        // Set circle color based on risk
        const isHighRisk = predictionClass.toLowerCase().includes('high') ||
                           probability > 0.5;
        
        if (isHighRisk) {
            circle.style.stroke = '#ef4444';
        } else {
            circle.style.stroke = '#10b981';
        }
    }
}

function displayInputSummary(inputSummary, timestamp) {
    if (!inputSummary) return;
    
    // Demographics
    const demographics = `${inputSummary.age} years, ${inputSummary.gender === 'M' ? 'Male' : 'Female'}`;
    document.getElementById('summary-demographics').textContent = demographics;
    
    // Risk factors
    const riskFactors = `${inputSummary.high_risk_factors} positive risk factors`;
    document.getElementById('summary-risk-factors').textContent = riskFactors;
    
    // Timestamp
    if (timestamp) {
        const date = new Date(timestamp);
        const timeString = date.toLocaleString();
        document.getElementById('summary-timestamp').textContent = timeString;
    }
}

// ==================== Reset Assessment ====================
function resetAssessment() {
    // Reset form
    assessmentForm.reset();
    
    // Reset step
    currentStep = 1;
    updateFormDisplay();
    updateFormNavigation();
    
    // Show assessment, hide results
    resultsSection.style.display = 'none';
    assessmentSection.style.display = 'block';
    
    // Scroll to assessment
    assessmentSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== Loading Overlay ====================
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// ==================== Notifications ====================
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        background: ${type === 'error' ? '#ef4444' : '#10b981'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        max-width: 400px;
    `;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'}" style="font-size: 1.25rem;"></i>
            <span style="font-weight: 500;">${message}</span>
        </div>
    `;
    
    // Add to body
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// ==================== Navbar Scroll Effect ====================
let lastScrollTop = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
    
    // Check if mobile menu is open
    const isMobileMenuOpen = navLinks && navLinks.classList.contains('active');
    
    // Add shadow when scrolled
    if (currentScroll > 50) {
        navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = 'none';
    }
    
    // Don't hide navbar if mobile menu is open
    if (isMobileMenuOpen) {
        navbar.style.transform = 'translateY(0)';
        lastScrollTop = currentScroll <= 0 ? 0 : currentScroll;
        return;
    }
    
    // Hide/show navbar based on scroll direction
    if (currentScroll <= 100) {
        // Always show navbar at the very top
        navbar.style.transform = 'translateY(0)';
    } else if (currentScroll > lastScrollTop && currentScroll > 100) {
        // Scrolling down - hide navbar
        navbar.style.transform = 'translateY(-100%)';
    } else if (currentScroll < lastScrollTop) {
        // Scrolling up - show navbar
        navbar.style.transform = 'translateY(0)';
    }
    
    lastScrollTop = currentScroll <= 0 ? 0 : currentScroll;
}, false);

// ==================== Intersection Observer for Animations ====================
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animatedElements = document.querySelectorAll('.process-card, .feature-card, .result-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });
});

// ==================== Prefill Demo Data (for testing) ====================
function prefillDemoData() {
    // Basic info
    document.querySelector('input[name="gender"][value="M"]').checked = true;
    document.getElementById('age').value = 65;
    
    // Risk factors
    document.querySelector('input[name="smoking"][value="2"]').checked = true;
    document.querySelector('input[name="yellow_fingers"][value="2"]').checked = true;
    document.querySelector('input[name="anxiety"][value="1"]').checked = true;
    document.querySelector('input[name="chronic_disease"][value="2"]').checked = true;
    document.querySelector('input[name="fatigue"][value="2"]').checked = true;
    document.querySelector('input[name="allergy"][value="1"]').checked = true;
    document.querySelector('input[name="alcohol_consuming"][value="1"]').checked = true;
    
    // Symptoms
    document.querySelector('input[name="wheezing"][value="2"]').checked = true;
    document.querySelector('input[name="coughing"][value="2"]').checked = true;
    document.querySelector('input[name="shortness_of_breath"][value="2"]').checked = true;
    document.querySelector('input[name="swallowing_difficulty"][value="1"]').checked = true;
    document.querySelector('input[name="chest_pain"][value="2"]').checked = true;
    document.querySelector('input[name="peer_pressure"][value="1"]').checked = true;
    
    validateCurrentStep();
}

// Add keyboard shortcut for demo data (Ctrl+Shift+D)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        prefillDemoData();
        showNotification('Demo data loaded!', 'info');
    }
});

console.log('%c🏥 LungAI - Lung Cancer Risk Prediction', 'color: #6366f1; font-size: 16px; font-weight: bold;');
console.log('%cPowered by Advanced Deep Learning & Random Forest Models', 'color: #6b7280; font-size: 12px;');
console.log('%cTip: Press Ctrl+Shift+D to load demo data', 'color: #10b981; font-size: 11px;');

