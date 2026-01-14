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
        loadDiagnosticReports();
        loadProcedures();
        loadDocumentReferences();
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
            <button class="tab" data-tab="diagnostics">Diagnostic Reports</button>
            <button class="tab" data-tab="procedures">Procedures</button>
            <button class="tab" data-tab="documents">Radiology Docs</button>
            <button class="tab" data-tab="debug">Debug Info</button>
        </div>
        <div id="conditions" class="tab-content active"><div class="card"><h2>Active Conditions</h2><div id="conditions-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="medications" class="tab-content"><div class="card"><h2>Current Medications</h2><div id="medications-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="allergies" class="tab-content"><div class="card"><h2>Allergies</h2><div id="allergies-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="vitals" class="tab-content"><div class="card"><h2>Recent Vitals</h2><div id="vitals-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="diagnostics" class="tab-content"><div class="card"><h2>Diagnostic Reports</h2><p class="section-description">Lab results, imaging, cardiology, and other diagnostic reports</p><div id="diagnostics-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="procedures" class="tab-content"><div class="card"><h2>Procedures</h2><p class="section-description">Surgeries, biopsies, endoscopies, and other performed procedures</p><div id="procedures-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
        <div id="documents" class="tab-content"><div class="card"><h2>Radiology Documents</h2><p class="section-description">Radiology results documentation with PDF references</p><div id="documents-list" class="loading"><div class="spinner"></div><p>Loading...</p></div></div></div>
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

// DiagnosticReport - Lab results, imaging, cardiology, endoscopy, audiology, EKG data
async function loadDiagnosticReports() {
    const container = document.getElementById('diagnostics-list');
    try {
        const bundle = await client.request(`DiagnosticReport?patient=${client.patient.id}&_count=50`);
        const reports = bundle.entry?.map(e => e.resource) || [];
        if (reports.length === 0) {
            container.innerHTML = '<div class="empty">No diagnostic reports found</div>';
            return;
        }
        container.innerHTML = reports.map(r => {
            const name = r.code?.text || r.code?.coding?.[0]?.display || 'Unknown Report';
            const status = r.status || 'unknown';
            const date = r.effectiveDateTime ? new Date(r.effectiveDateTime).toLocaleDateString() :
                         r.issued ? new Date(r.issued).toLocaleDateString() : 'Date unknown';
            const category = r.category?.[0]?.coding?.[0]?.display || r.category?.[0]?.text || '';
            const conclusion = r.conclusion ? `<br><em>${r.conclusion.substring(0, 150)}${r.conclusion.length > 150 ? '...' : ''}</em>` : '';
            const statusClass = status === 'final' ? '' : status === 'preliminary' ? 'warning' : '';
            return `<div class="list-item ${statusClass}">
                <strong>${name}</strong>
                <small>${category ? category + ' | ' : ''}Status: ${status} | Date: ${date}</small>
                ${conclusion}
            </div>`;
        }).join('');
    } catch (error) {
        console.error("Error loading diagnostic reports:", error);
        container.innerHTML = `<div class="error">Error loading diagnostic reports: ${error.message}</div>`;
    }
}

// Procedure - Surgeries, biopsies, endoscopies, counseling, physiotherapy
async function loadProcedures() {
    const container = document.getElementById('procedures-list');
    try {
        const bundle = await client.request(`Procedure?patient=${client.patient.id}&_count=50`);
        const procedures = bundle.entry?.map(e => e.resource) || [];
        if (procedures.length === 0) {
            container.innerHTML = '<div class="empty">No procedures found</div>';
            return;
        }
        container.innerHTML = procedures.map(p => {
            const name = p.code?.text || p.code?.coding?.[0]?.display || 'Unknown Procedure';
            const status = p.status || 'unknown';
            const performedDate = p.performedDateTime ? new Date(p.performedDateTime).toLocaleDateString() :
                                  p.performedPeriod?.start ? new Date(p.performedPeriod.start).toLocaleDateString() : 'Date unknown';
            const category = p.category?.coding?.[0]?.display || p.category?.text || '';
            const performer = p.performer?.[0]?.actor?.display || '';
            const location = p.location?.display || '';
            const statusClass = status === 'completed' ? '' : status === 'in-progress' ? 'warning' : '';
            let details = [];
            if (category) details.push(category);
            if (performer) details.push(`Performed by: ${performer}`);
            if (location) details.push(`Location: ${location}`);
            return `<div class="list-item ${statusClass}">
                <strong>${name}</strong>
                <small>Status: ${status} | Date: ${performedDate}</small>
                ${details.length > 0 ? `<small>${details.join(' | ')}</small>` : ''}
            </div>`;
        }).join('');
    } catch (error) {
        console.error("Error loading procedures:", error);
        container.innerHTML = `<div class="error">Error loading procedures: ${error.message}</div>`;
    }
}

// DocumentReference - Radiology results documentation (per Epic FHIR R4 API)
async function loadDocumentReferences() {
    const container = document.getElementById('documents-list');
    try {
        // Epic's DocumentReference (Radiology Results) API - returns radiology result documentation
        // including references to Binary resources containing PDFs
        const bundle = await client.request(`DocumentReference?patient=${client.patient.id}&_count=50`);
        const documents = bundle.entry?.map(e => e.resource) || [];
        if (documents.length === 0) {
            container.innerHTML = '<div class="empty">No radiology documents found</div>';
            return;
        }
        container.innerHTML = documents.map(d => {
            const description = d.description || d.type?.text || d.type?.coding?.[0]?.display || 'Unknown Document';
            const status = d.status || 'unknown';
            const date = d.date ? new Date(d.date).toLocaleDateString() :
                         d.context?.period?.start ? new Date(d.context.period.start).toLocaleDateString() : 'Date unknown';
            const category = d.category?.[0]?.coding?.[0]?.display || d.category?.[0]?.text || '';
            const author = d.author?.[0]?.display || '';
            const contentType = d.content?.[0]?.attachment?.contentType || '';
            // Check if there's a Binary reference for the PDF
            const binaryUrl = d.content?.[0]?.attachment?.url || '';
            const hasPdf = binaryUrl.includes('Binary') || contentType === 'application/pdf';
            const statusClass = status === 'current' ? '' : 'warning';
            let details = [];
            if (category) details.push(category);
            if (author) details.push(`Author: ${author}`);
            if (contentType) details.push(`Format: ${contentType}`);
            if (hasPdf) details.push('PDF Available');
            return `<div class="list-item ${statusClass}">
                <strong>${description}</strong>
                <small>Status: ${status} | Date: ${date}</small>
                ${details.length > 0 ? `<small>${details.join(' | ')}</small>` : ''}
            </div>`;
        }).join('');
    } catch (error) {
        console.error("Error loading document references:", error);
        container.innerHTML = `<div class="error">Error loading radiology documents: ${error.message}</div>`;
    }
}