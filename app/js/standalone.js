let selectedEHR = null;

document.getElementById('ehrSelect').addEventListener('change', function() {
    selectedEHR = this.value;
    const customForm = document.getElementById('customForm');
    if (selectedEHR === 'custom') customForm.classList.add('show');
    else customForm.classList.remove('show');
    document.getElementById('launchBtn').disabled = !selectedEHR;
});

document.getElementById('launchBtn').addEventListener('click', function() {
    if (!selectedEHR) { showError('Please select an EHR system'); return; }
    if (selectedEHR === 'custom') {
        const fhirUrl = document.getElementById('customFhirUrl').value;
        const clientId = document.getElementById('customClientId').value;
        if (!fhirUrl || !clientId) { showError('Please fill in all custom server fields'); return; }
        EHR_CONFIGS.custom.fhirUrl = fhirUrl;
        EHR_CONFIGS.custom.clientId = clientId;
    }
    const config = getEHRConfig(selectedEHR);
    if (!config.clientId && selectedEHR !== 'custom') {
        showError(`${config.name} client ID is not configured.`);
        return;
    }
    launchSMART(selectedEHR);
});

function showError(message) {
    const errorBox = document.getElementById('errorBox');
    document.getElementById('errorMessage').textContent = message;
    errorBox.classList.add('show');
}

function launchSMART(ehrKey) {
    try {
        const smartConfig = getSMARTConfig(ehrKey, true);
        console.log('Launching SMART with config:', smartConfig);
        FHIR.oauth2.authorize(smartConfig).catch(error => {
            console.error('Authorization error:', error);
            showError(`Authorization failed: ${error.message || 'Unknown error'}`);
        });
    } catch (error) {
        console.error('Launch error:', error);
        showError(`Failed to launch: ${error.message || 'Unknown error'}`);
    }
}