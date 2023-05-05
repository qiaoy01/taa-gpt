// Fetch the list of available config files and project files
async function fetchConfigList() {
    const response = await fetch('/config_list');
    const data = await response.json();
    return {
        config_list: data.config_list,
        project_list: data.project_list
    };
}

// Populate the config select dropdown and project select dropdown
async function populateSelects() {
    const lists = await fetchConfigList();
    const configList = lists.config_list;
    const projectList = lists.project_list;
    const configSelect = document.getElementById('configSelect');
    const projectSelect = document.getElementById('projectSelect');

    configList.forEach(config => {
        const configName = config.replace('config_', '').replace('.json', '');
        const option = document.createElement('option');
        option.value = configName;
        option.innerText = configName;
        configSelect.appendChild(option);
    });

    if (projectSelect) {
        projectList.forEach(project => {
            const projectName = project.replace('train_', '').replace('.py', '');
            const option = document.createElement('option');
            option.value = projectName;
            option.innerText = projectName;
            projectSelect.appendChild(option);
        });
    } else {
        console.error('Project select element not found');
    }

    const defaultOptionIndex = Array.from(configSelect.options).findIndex(option => option.value === 'default');
    if (defaultOptionIndex !== -1) {
        configSelect.selectedIndex = defaultOptionIndex;
        onConfigSelectChange();
    }
}

async function startTraining() {
    const projectSelect = document.getElementById('projectSelect');
    const selectedProject = projectSelect.value;
    const configDetails = document.getElementById('configDetails').value;

    console.log('selectedProject:', selectedProject);
    console.log('configDetails:', configDetails);

    const payload = {
        program: selectedProject,
        config: configDetails
    };

    const response = await fetch('/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (response.ok) {
        alert('Training started successfully.');
    } else {
        alert('Error starting training. Please check the logs for more information.');
    }
}

// Fetch the config details for the selected config
async function fetchConfigDetails(configName) {
    const response = await fetch(`/config_details?name=${configName}`);
    const data = await response.json();
    return data;
}

// Update the config details textarea and model size info when a new config is selected
async function updateConfigDetails() {
    
    const configSelect = document.getElementById('configSelect');
    const selectedConfigName = configSelect.value;

    const configData = await fetchConfigDetails(selectedConfigName);
    const configDetailsTextarea = document.getElementById('configDetails');
    configDetailsTextarea.value = JSON.stringify(configData.config, null, 2);

    document.getElementById('modelSize').innerText = configData.model_size;
    document.getElementById('modelSizeM').innerText = configData.model_size_m;
    document.getElementById('modelSizeG').innerText = configData.total_size_g;
}

async function onConfigSelectChange() {
    const configSelect = document.getElementById('configSelect');
    const selectedConfigName = configSelect.value;
    const configDetails = await fetchConfigDetails(selectedConfigName);

    // Update the config details display here with the fetched configDetails
    updateConfigDetails();
}

async function saveConfig() {
    const configSelect = document.getElementById('configSelect');
    const configName = configSelect.value;

    if (configName === 'default') {
        alert("You cannot overwrite the default config.");
        return;
    }

    const configDetails = document.getElementById('configDetails').value;
    await fetch(`/save_model_config?name=${configName}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: configDetails
    });

    alert('Config saved successfully.');
}

async function saveConfigAs() {
    const configName = prompt("Enter a new config name. It can only contain letters, numbers, underscores, and dots.");

    if (!configName || !/^[\w\.]+$/.test(configName) || configName.includes('config') || configName.includes('json')) {
        alert("Invalid config name. It can only contain letters, numbers, underscores, and dots, and should not include 'config' or 'json'.");
        return;
    }

    const configDetails = document.getElementById('configDetails').value;
    await fetch(`/save_model_config?name=${configName}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: configDetails
    });

    alert('Config saved successfully.');
}

// Add event listener for the config select dropdown
document.getElementById('configSelect').addEventListener('change', updateConfigDetails);
document.getElementById('executeTrainingBtn').addEventListener('click', startTraining);

// Initialize the dropdown and config details
populateSelects().then(() => {
    updateConfigDetails();
});

