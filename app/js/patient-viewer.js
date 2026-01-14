let client = null;

FHIR.oauth2.ready()
    .then(function(smartClient) {
        client = smartClient;
        console.log("SMART client ready:", client);
        updateEHRBadge();
        displayUserInfo();
        return loadPatientData();
    })
    .catch(function(error) {
        console.error("SMART initialization error:", error);
        document.getElementById('content').innerHTML = `
            <div class="card">
                <div class="error">
                    <strong>Connection Failed</strong>
                    <p>${error.message || 'Failed to connect to EHR'}</p>
                    <p style="font-size: 12px;">Make sure you launched the app correctly from your EHR system or use the standalone launch page.</p>
                </div>
            </div>
        `;
    });

function updateEHRBadge() {
    const badge = document.getElementById('ehr-badge');
    const serverUrl = client.state.serverUrl || '';
    if (serverUrl.includes('epic.com')) badge.textContent = 'Connected to Epic';
    else if (serverUrl.includes('cerner.com')) badge.textContent = 'Connected to Cerner';
    else if (serverUrl.includes('openemr')) badge.textContent = 'Connected to OpenEMR';
    else badge.textContent = 'Connected to FHIR Server';
}

function displayUserInfo() {
    const userInfo = document.getElementById('user-info');
    if (client.user && client.user.fhirUser) {
        client.request(client.user.fhirUser)
            .then(function(user) {
                const name = user.name ? (user.name[0].given?.join(' ') + ' ' + user.name[0].family) : 'Unknown User';
                userInfo.textContent = `Logged in as: ${name}`;
            })
            .catch(function() { userInfo.textContent = 'User: ' + (client.user.fhirUser || 'Unknown'); });
    }
}

async function loadPatientData() {
    try {
        const patient = await client.patient.read();
        console.log("Patient:", patient);
        let html = buildPatientCard(patient);
        html += buildTabsUI();
        document.getElementById('content').innerHTML = html;
        setupTabs();
        loadConditions();
        loadMedications();
        loadAllergies();
        loadObservations();
    } catch (error) {
        console.error("Error loading patient:", error);
        document.getElementById('content').innerHTML = `<div class="card"><div class="error"><strong>Error Loading Patient</strong><p>${error.message}</p></div></div>`;
    }
}

function buildPatientCard(patient) {
    const name = patient.name?.[0];
    const fullName = name ? `${name.given?.join(' ') || ''} ${name.family || ''}`.trim() : 'Unknown';
    const initials = fullName.split(' ').map(n => n[0]).join('').substring(0, 2);
    const birthDate = patient.birthDate || 'Unknown';
    const gender = patient.gender ? patient.gender.charAt(0).toUpperCase() + patient.gender.slice(1) : 'Unknown';
    let age = 'Unknown';
    if (patient.birthDate) {
        const birth = new Date(patient.birthDate);
        age = Math.floor((new Date() - birth) / (365.25 * 24 * 60 * 60 * 1000));
    }
    let mrn = 'N/A';
    if (patient.identifier) {
        const mrnId = patient.identifier.find(id => id.type?.text === 'MRN' || id.type?.coding?.some(c => c.code === 'MR'));
        if (mrnId) mrn = mrnId.value;
    }
    let address = 'Not available';
    if (patient.address?.[0]) {
        const addr = patient.address[0];
        address = [addr.line?.join(', '), addr.city, addr.state, addr.postalCode].filter(Boolean).join(', ');
    }
    let phone = 'Not available';
    if (patient.telecom) {
        const phoneContact = patient.telecom.find(t => t.system === 'phone');
        if (phoneContact) phone = phoneContact.value;
    }
    return `
        <div class="card">
            <div class="patient-header">
                <div class="patient-avatar">${initials}</div>
                <div class="patient-info">
                    <h3>${fullName}</h3>
                    <p>MRN: ${mrn} | FHIR ID: ${patient.id}</p>
                </div>
            </div>
            <div class="info-grid">
                <div class="info-item"><label>Date of Birth</label><span>${birthDate}</span></div>
                <div class="info-item"><label>Age</label><span>${age} years</span></div>
                <div class="info-item"><label>Gender</label><span>${gender}</span></div>
                <div class="info-item"><label>Phone</label><span>${phone}</span></div>
                <div class="info-item" style="grid-column: span 2;"><label>Address</label><span>${address}</span></div>
            </div>
        </div>
    `;
}

function buildTabsUI() {
    return `
        <div class="tabs">
            <button class="tab active" data-tab="conditions">Conditions</button>
            <button class="tab" data-tab="medications">Medications</button>
            <button class="tab" data-tab="allergies">Allergies</button>
            <button class="tab" data-tab="vitals">Vitals</button>
            <button class="tab" data-tab="debug">Debug Info</button>
        </div>
        <div id="conditions" class="tab-content active"><div class="card"><h2>Active Conditions</h2><div id="conditions-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="medications" class="tab-content"><div class="card"><h2>Current Medications</h2><div id="medications-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="allergies" class="tab-content"><div class="card"><h2>Allergies</h2><div id="allergies-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="vitals" class="tab-content"><div class="card"><h2>Recent Vitals</h2><div id="vitals-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="debug" class="tab-content"><div class="card"><h2>Debug Information</h2><div class="debug-section" id="debug-info">Loading...</div></div></div>
    `;
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(this.dataset.tab).classList.add('active');
        });
    });
    loadDebugInfo();
}

async function loadConditions() {
    const container = document.getElementById('conditions-list');
    try {
        const bundle = await client.request(`Condition?patient=${client.patient.id}`);
        const conditions = bundle.entry?.map(e => e.resource) || [];
        if (conditions.length === 0) { container.innerHTML = '<div class="empty">No conditions found</div>'; return; }
        container.innerHTML = conditions.map(c => {
            const name = c.code?.text || c.code?.coding?.[0]?.display || 'Unknown';
            const date = c.recordedDate ? new Date(c.recordedDate).toLocaleDateString() : 'Date unknown';
            const status = c.clinicalStatus?.coding?.[0]?.code || '';
            return `<div class="list-item"><strong>${name}</strong><small>Recorded: ${date} ${status ? '| Status: ' + status : ''}</small></div>`;
        }).join('');
    } catch (error) { container.innerHTML = `<div class="error">Error: ${error.message}</div>`; }
}

async function loadMedications() {
    const container = document.getElementById('medications-list');
    try {
        const bundle = await client.request(`MedicationRequest?patient=${client.patient.id}&status=active`);
        const meds = bundle.entry?.map(e => e.resource) || [];
        if (meds.length === 0) { container.innerHTML = '<div class="empty">No active medications</div>'; return; }
        container.innerHTML = meds.map(m => {
            let name = m.medicationCodeableConcept?.text || m.medicationCodeableConcept?.coding?.[0]?.display || m.medicationReference?.display || 'Unknown';
            const dosage = m.dosageInstruction?.[0]?.text || 'No dosage info';
            return `<div class="list-item"><strong>${name}</strong><small>${dosage}</small></div>`;
        }).join('');
    } catch (error) { container.innerHTML = `<div class="error">Error: ${error.message}</div>`; }
}

async function loadAllergies() {
    const container = document.getElementById('allergies-list');
    try {
        const bundle = await client.request(`AllergyIntolerance?patient=${client.patient.id}`);
        const allergies = bundle.entry?.map(e => e.resource) || [];
        if (allergies.length === 0) { container.innerHTML = '<div class="empty">No allergies recorded</div>'; return; }
        container.innerHTML = allergies.map(a => {
            const name = a.code?.text || a.code?.coding?.[0]?.display || 'Unknown';
            const criticality = a.criticality || 'Unknown';
            const cssClass = criticality === 'high' ? 'danger' : criticality === 'low' ? '' : 'warning';
            return `<div class="list-item ${cssClass}"><strong>${name}</strong><small>Criticality: ${criticality}</small></div>`;
        }).join('');
    } catch (error) { container.innerHTML = `<div class="error">Error: ${error.message}</div>`; }
}

async function loadObservations() {
    const container = document.getElementById('vitals-list');
    try {
        let bundle;
        try { bundle = await client.request(`Observation?patient=${client.patient.id}&category=vital-signs&_count=20`); }
        catch { bundle = await client.request(`Observation?patient=${client.patient.id}&_count=20`); }
        const obs = bundle.entry?.map(e => e.resource) || [];
        if (obs.length === 0) { container.innerHTML = '<div class="empty">No observations found</div>'; return; }
        container.innerHTML = obs.map(o => {
            const name = o.code?.text || o.code?.coding?.[0]?.display || 'Unknown';
            let value = 'No value';
            if (o.valueQuantity) value = `${o.valueQuantity.value} ${o.valueQuantity.unit || ''}`;
            else if (o.valueCodeableConcept) value = o.valueCodeableConcept.text || o.valueCodeableConcept.coding?.[0]?.display;
            const date = o.effectiveDateTime ? new Date(o.effectiveDateTime).toLocaleString() : '';
            return `<div class="list-item"><strong>${name}: ${value}</strong><small>${date}</small></div>`;
        }).join('');
    } catch (error) { container.innerHTML = `<div class="error">Error: ${error.message}</div>`; }
}

function loadDebugInfo() {
    document.getElementById('debug-info').textContent = JSON.stringify({
        "FHIR Server": client.state.serverUrl,
        "Patient ID": client.patient.id,
        "Scope": client.state.tokenResponse?.scope,
        "Client ID": client.state.clientId
    }, null, 2);
}