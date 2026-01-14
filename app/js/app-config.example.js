const EHR_CONFIGS = {
    epic: {
        name: "Epic",
        clientId: "YOUR_EPIC_CLIENT_ID",
        authUrl: "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/authorize",
        tokenUrl: "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token",
        fhirUrl: "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4/",
        redirectUri: "http://localhost:3000/app/index.html",
        scope: "YOUR_EHR_LAUNCH_SCOPES",
        standaloneScope: "YOUR_STANDALONE_SCOPES",
        issPattern: /fhir\.epic\.com/i
    },

    cerner: {
        name: "Cerner",
        clientId: "YOUR_CERNER_CLIENT_ID",
        authUrl: "https://authorization.cerner.com/tenants/YOUR_TENANT_ID/protocols/oauth2/profiles/smart-v1/personas/provider/authorize",
        tokenUrl: "https://authorization.cerner.com/tenants/YOUR_TENANT_ID/protocols/oauth2/profiles/smart-v1/token",
        fhirUrl: "https://fhir-ehr-code.cerner.com/r4/YOUR_TENANT_ID/",
        redirectUri: "http://localhost:3000/app/index.html",
        scope: "YOUR_EHR_LAUNCH_SCOPES",
        standaloneScope: "YOUR_STANDALONE_SCOPES",
        issPattern: /cerner\.com/i
    },

    openemr: {
        name: "OpenEMR",
        clientId: "",
        authUrl: "https://your-openemr-instance.com/oauth2/default/authorize",
        tokenUrl: "https://your-openemr-instance.com/oauth2/default/token",
        fhirUrl: "https://your-openemr-instance.com/apis/default/fhir/",
        redirectUri: "http://localhost:3000/app/index.html",
        scope: "YOUR_EHR_LAUNCH_SCOPES",
        standaloneScope: "YOUR_STANDALONE_SCOPES",
        issPattern: /openemr/i
    },

    custom: {
        name: "Custom FHIR Server",
        clientId: "",
        authUrl: "",
        tokenUrl: "",
        fhirUrl: "",
        redirectUri: "http://localhost:3000/app/index.html",
        scope: "YOUR_EHR_LAUNCH_SCOPES",
        standaloneScope: "YOUR_STANDALONE_SCOPES",
        issPattern: null
    }
};

function detectEHR(iss) {
    if (!iss) return 'custom';
    for (const [key, config] of Object.entries(EHR_CONFIGS)) {
        if (config.issPattern && config.issPattern.test(iss)) return key;
    }
    return 'custom';
}

function getEHRConfig(ehrKey) {
    return EHR_CONFIGS[ehrKey] || EHR_CONFIGS.custom;
}

function getSMARTConfig(ehrKey, isStandalone = false) {
    const config = getEHRConfig(ehrKey);
    const smartConfig = {
        clientId: config.clientId,
        scope: isStandalone ? config.standaloneScope : config.scope,
        redirectUri: config.redirectUri
    };
    if (ehrKey === 'cerner') {
        smartConfig.fhirServiceUrl = config.fhirUrl;
        smartConfig.authorizeUri = config.authUrl;
        smartConfig.tokenUri = config.tokenUrl;
    } else if (isStandalone) {
        smartConfig.iss = config.fhirUrl;
    }
    return smartConfig;
}