const BASE_URL = 'http://localhost:5000';

async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${BASE_URL}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API fetch error:', error);
        throw error;
    }
}

function animateCounter(element, targetValue, duration = 2000) {
    const startValue = 0;
    const startTime = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = Math.floor(startValue + (targetValue - startValue) * progress);
        element.textContent = currentValue.toLocaleString('en-IN');
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading-spinner"></div>';
    }
}

function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        const spinner = container.querySelector('.loading-spinner');
        if (spinner) {
            spinner.remove();
        }
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'card fade-in';
    errorDiv.style.borderLeft = '4px solid #e94560';
    errorDiv.innerHTML = `
        <h3 style="color: #e94560; margin-bottom: 10px;">⚠️ Error</h3>
        <p>${message}</p>
    `;
    
    const container = document.querySelector('.container') || document.body;
    container.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function formatCurrency(amount) {
    return '₹' + amount.toLocaleString('en-IN');
}

function formatPercentage(value) {
    return value.toFixed(1) + '%';
}

// Utility function to create chart configurations
function createChartConfig(type, data, options = {}) {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#eaeaea',
                    font: {
                        family: 'Inter'
                    }
                }
            }
        },
        scales: type !== 'doughnut' && type !== 'radar' ? {
            x: {
                grid: {
                    color: '#2a2a4a'
                },
                ticks: {
                    color: '#eaeaea'
                }
            },
            y: {
                grid: {
                    color: '#2a2a4a'
                },
                ticks: {
                    color: '#eaeaea'
                }
            }
        } : {}
    };
    
    return {
        type: type,
        data: data,
        options: { ...defaultOptions, ...options }
    };
}

// Initialize page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
});