/**
 * JobApp Helper — Main Frontend Application Logic
 * Supports SSE streaming, vector discovery, dynamic candidate profiles, and outreach editing.
 */

// Global State
const state = {
  activeTab: 'generator',
  mode: 'single', // 'single' | 'batch'
  profile: null,
  profilesList: [],
  settings: {
    provider: 'gemini',
    model: 'gemini-3.8-flash',
    apiKey: '',
  },
  currentApplicationSlug: null,
  archiveFilter: 'all',
  isGenerating: false,
};

// DOM References
const elements = {
  // Navigation
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  memoryCountBadge: document.getElementById('memoryCountBadge'),
  archiveCountBadge: document.getElementById('archiveCountBadge'),
  
  // Profile Pill
  profilePillBtn: document.getElementById('profilePillBtn'),
  avatarInitial: document.getElementById('avatarInitial'),
  pillCandidateName: document.getElementById('pillCandidateName'),
  pillTargetRole: document.getElementById('pillTargetRole'),
  
  // Generator Inputs
  modeSingleBtn: document.getElementById('modeSingleBtn'),
  modeBatchBtn: document.getElementById('modeBatchBtn'),
  singleFormBox: document.getElementById('singleFormBox'),
  batchFormBox: document.getElementById('batchFormBox'),
  targetStartupInput: document.getElementById('targetStartupInput'),
  forceRefreshCheckbox: document.getElementById('forceRefreshCheckbox'),
  runPipelineBtn: document.getElementById('runPipelineBtn'),
  batchStartupsTextarea: document.getElementById('batchStartupsTextarea'),
  runBatchBtn: document.getElementById('runBatchBtn'),
  chipSuggests: document.querySelectorAll('.chip-suggest'),

  // Pipeline Tracker
  pipelineTrackerCard: document.getElementById('pipelineTrackerCard'),
  trackerTitle: document.getElementById('trackerTitle'),
  trackerSubtitle: document.getElementById('trackerSubtitle'),
  trackerProgressNum: document.getElementById('trackerProgressNum'),
  progressBarFill: document.getElementById('progressBarFill'),
  stepNodes: [
    document.getElementById('stepNode1'),
    document.getElementById('stepNode2'),
    document.getElementById('stepNode3'),
    document.getElementById('stepNode4'),
    document.getElementById('stepNode5'),
  ],
  logsFeed: document.getElementById('logsFeed'),
  toggleLogsBtn: document.getElementById('toggleLogsBtn'),

  // Outreach Studio Workspace
  outreachStudioCard: document.getElementById('outreachStudioCard'),
  studioStartupInitial: document.getElementById('studioStartupInitial'),
  studioStartupName: document.getElementById('studioStartupName'),
  studioSourceBadge: document.getElementById('studioSourceBadge'),
  studioDateBadge: document.getElementById('studioDateBadge'),
  studioTargetRoleDisplay: document.getElementById('studioTargetRoleDisplay'),
  studioFitScore: document.getElementById('studioFitScore'),
  studioFitLabel: document.getElementById('studioFitLabel'),
  studioFitHint: document.getElementById('studioFitHint'),
  studioCompanySummary: document.getElementById('studioCompanySummary'),
  studioTechChips: document.getElementById('studioTechChips'),
  studioFitReasoning: document.getElementById('studioFitReasoning'),
  studioMatchingSkills: document.getElementById('studioMatchingSkills'),
  studioSkillGaps: document.getElementById('studioSkillGaps'),
  studioEmailSubject: document.getElementById('studioEmailSubject'),
  studioColdEmail: document.getElementById('studioColdEmail'),
  studioLinkedinNote: document.getElementById('studioLinkedinNote'),
  linkedinCharCount: document.getElementById('linkedinCharCount'),
  copyEmailBtn: document.getElementById('copyEmailBtn'),
  openMailtoBtn: document.getElementById('openMailtoBtn'),
  copyLinkedinBtn: document.getElementById('copyLinkedinBtn'),
  saveDraftEditsBtn: document.getElementById('saveDraftEditsBtn'),

  // Discovery Tab
  discoveryTotalCount: document.getElementById('discoveryTotalCount'),
  discoveryQueryInput: document.getElementById('discoveryQueryInput'),
  discoverySearchBtn: document.getElementById('discoverySearchBtn'),
  discoveryResultsTitle: document.getElementById('discoveryResultsTitle'),
  discoveryResultsCount: document.getElementById('discoveryResultsCount'),
  discoveryResultsGrid: document.getElementById('discoveryResultsGrid'),
  chipSuggestDiscs: document.querySelectorAll('.chip-suggest-disc'),

  // Archive Tab
  metricTotalApps: document.getElementById('metricTotalApps'),
  metricHighFit: document.getElementById('metricHighFit'),
  metricMedFit: document.getElementById('metricMedFit'),
  metricSkipped: document.getElementById('metricSkipped'),
  archiveSearchInput: document.getElementById('archiveSearchInput'),
  archiveFilterPills: document.querySelectorAll('.filter-pill'),
  archiveGrid: document.getElementById('archiveGrid'),

  // Modals
  profileModal: document.getElementById('profileModal'),
  closeProfileModalBtn: document.getElementById('closeProfileModalBtn'),
  cancelProfileBtn: document.getElementById('cancelProfileBtn'),
  saveProfileBtn: document.getElementById('saveProfileBtn'),
  profileForm: document.getElementById('profileForm'),

  settingsModal: document.getElementById('settingsModal'),
  openSettingsBtn: document.getElementById('openSettingsBtn'),
  closeSettingsModalBtn: document.getElementById('closeSettingsModalBtn'),
  cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  providerGeminiCard: document.getElementById('providerGeminiCard'),
  providerGroqCard: document.getElementById('providerGroqCard'),
  settingModel: document.getElementById('settingModel'),
  settingApiKey: document.getElementById('settingApiKey'),
  testConnectionBtn: document.getElementById('testConnectionBtn'),
  testResultMsg: document.getElementById('testResultMsg'),

  toastContainer: document.getElementById('toastContainer'),
};

// ---------------------------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  loadSavedSettings();
  setupEventListeners();
  await loadCandidateProfiles();
  await refreshSystemStats();
  await loadArchiveApplications();
  await loadDiscoveryItems('AI startup product platform');
});

function loadSavedSettings() {
  const saved = localStorage.getItem('jobapp_settings');
  if (saved) {
    try {
      state.settings = { ...state.settings, ...JSON.parse(saved) };
    } catch (e) {
      console.warn('Failed to parse settings', e);
    }
  }
}

function saveSettingsToStorage() {
  localStorage.setItem('jobapp_settings', JSON.stringify(state.settings));
}

// ---------------------------------------------------------------------------
// EVENT LISTENERS
// ---------------------------------------------------------------------------

function setupEventListeners() {
  // Navigation
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Mode Selector (Single vs Batch)
  elements.modeSingleBtn.addEventListener('click', () => setMode('single'));
  elements.modeBatchBtn.addEventListener('click', () => setMode('batch'));

  // Suggestions chips
  elements.chipSuggests.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.targetStartupInput.value = chip.dataset.company;
      elements.targetStartupInput.focus();
    });
  });

  // Run Pipeline Button
  elements.runPipelineBtn.addEventListener('click', () => {
    const startup = elements.targetStartupInput.value.trim();
    if (!startup) {
      showToast('Please enter a startup name.', 'error');
      elements.targetStartupInput.focus();
      return;
    }
    runSinglePipeline(startup);
  });

  elements.targetStartupInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') elements.runPipelineBtn.click();
  });

  // Run Batch Button
  elements.runBatchBtn.addEventListener('click', () => {
    const text = elements.batchStartupsTextarea.value.trim();
    if (!text) {
      showToast('Please enter one or more startup names.', 'error');
      return;
    }
    const list = text.split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
    if (list.length === 0) {
      showToast('No valid startup names found.', 'error');
      return;
    }
    runBatchPipeline(list);
  });

  // Toggle Log Feed
  elements.toggleLogsBtn.addEventListener('click', () => {
    const isHidden = elements.logsFeed.classList.contains('hidden');
    elements.logsFeed.classList.toggle('hidden', !isHidden);
  });

  // Copy Buttons
  elements.copyEmailBtn.addEventListener('click', () => {
    const text = elements.studioColdEmail.value;
    navigator.clipboard.writeText(text);
    showToast('Cold email copied to clipboard!', 'success');
  });

  elements.copyLinkedinBtn.addEventListener('click', () => {
    const text = elements.studioLinkedinNote.value;
    navigator.clipboard.writeText(text);
    showToast('LinkedIn note copied to clipboard!', 'success');
  });

  // Real-time LinkedIn Char Counter
  elements.studioLinkedinNote.addEventListener('input', () => {
    updateLinkedinCharCount();
  });

  // Save Outreach Edits
  elements.saveDraftEditsBtn.addEventListener('click', async () => {
    if (!state.currentApplicationSlug) {
      showToast('No active application to save.', 'error');
      return;
    }
    try {
      const res = await fetch(`/api/applications/${state.currentApplicationSlug}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cold_email: elements.studioColdEmail.value,
          email_subject: elements.studioEmailSubject.value,
          linkedin_note: elements.studioLinkedinNote.value,
        }),
      });
      if (!res.ok) throw new Error('Save failed');
      showToast('Edits saved to local applications record!', 'success');
      await loadArchiveApplications();
    } catch (e) {
      showToast(e.message, 'error');
    }
  });

  // Discovery Tab Search
  elements.discoverySearchBtn.addEventListener('click', () => {
    const query = elements.discoveryQueryInput.value.trim();
    if (query) loadDiscoveryItems(query);
  });

  elements.discoveryQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') elements.discoverySearchBtn.click();
  });

  elements.chipSuggestDiscs.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.discoveryQueryInput.value = chip.dataset.query;
      loadDiscoveryItems(chip.dataset.query);
    });
  });

  // Archive Filter Pills
  elements.archiveFilterPills.forEach(pill => {
    pill.addEventListener('click', () => {
      elements.archiveFilterPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.archiveFilter = pill.dataset.filter;
      loadArchiveApplications();
    });
  });

  // Archive Search Input
  let searchTimeout;
  elements.archiveSearchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(loadArchiveApplications, 300);
  });

  // Profile Modal
  elements.profilePillBtn.addEventListener('click', openProfileModal);
  elements.closeProfileModalBtn.addEventListener('click', closeProfileModal);
  elements.cancelProfileBtn.addEventListener('click', closeProfileModal);
  elements.saveProfileBtn.addEventListener('click', saveCandidateProfile);

  // Settings Modal
  elements.openSettingsBtn.addEventListener('click', openSettingsModal);
  elements.closeSettingsModalBtn.addEventListener('click', closeSettingsModal);
  elements.cancelSettingsBtn.addEventListener('click', closeSettingsModal);
  elements.saveSettingsBtn.addEventListener('click', saveSettingsPreferences);

  elements.providerGeminiCard.addEventListener('click', () => {
    elements.providerGeminiCard.classList.add('active');
    elements.providerGroqCard.classList.remove('active');
    elements.providerGeminiCard.querySelector('input').checked = true;
    populateModelDropdown('gemini');
  });

  elements.providerGroqCard.addEventListener('click', () => {
    elements.providerGroqCard.classList.add('active');
    elements.providerGeminiCard.classList.remove('active');
    elements.providerGroqCard.querySelector('input').checked = true;
    populateModelDropdown('groq');
  });

  elements.testConnectionBtn.addEventListener('click', testAPIConnection);
}

// ---------------------------------------------------------------------------
// TAB & MODE SWITCHING
// ---------------------------------------------------------------------------

function switchTab(tabId) {
  state.activeTab = tabId;
  elements.navTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabId);
  });
  elements.tabPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab${tabId.charAt(0).toUpperCase() + tabId.slice(1)}`);
  });

  if (tabId === 'archive') {
    loadArchiveApplications();
  } else if (tabId === 'discovery') {
    refreshSystemStats();
  }
}

function setMode(mode) {
  state.mode = mode;
  elements.modeSingleBtn.classList.toggle('active', mode === 'single');
  elements.modeBatchBtn.classList.toggle('active', mode === 'batch');
  elements.singleFormBox.classList.toggle('hidden', mode !== 'single');
  elements.batchFormBox.classList.toggle('hidden', mode !== 'batch');
}

// ---------------------------------------------------------------------------
// PROFILE MANAGEMENT
// ---------------------------------------------------------------------------

async function loadCandidateProfiles() {
  try {
    const res = await fetch('/api/profiles');
    if (!res.ok) throw new Error('Could not fetch profiles');
    const profiles = await res.json();
    state.profilesList = profiles;
    state.profile = profiles[0] || null;
    updateProfilePillUI();
  } catch (e) {
    console.error('Failed to load profile', e);
  }
}

function updateProfilePillUI() {
  if (!state.profile) return;
  const name = state.profile.name || 'Candidate';
  elements.avatarInitial.textContent = name.charAt(0).toUpperCase();
  elements.pillCandidateName.textContent = name;
  elements.pillTargetRole.textContent = state.profile.target_role || 'Candidate';
}

function openProfileModal() {
  if (!state.profile) return;
  document.getElementById('profName').value = state.profile.name || '';
  document.getElementById('profTargetRole').value = state.profile.target_role || '';
  document.getElementById('profBackground').value = state.profile.background || '';
  document.getElementById('profSkills').value = state.profile.skills || '';
  document.getElementById('profProjects').value = state.profile.projects || '';
  document.getElementById('profGithub').value = state.profile.github || '';
  document.getElementById('profLinkedin').value = state.profile.linkedin || '';
  document.getElementById('profEmail').value = state.profile.contact_email || '';
  document.getElementById('profTone').value = state.profile.tone || 'confident, concise, and technically grounded';

  elements.profileModal.classList.remove('hidden');
}

function closeProfileModal() {
  elements.profileModal.classList.add('hidden');
}

async function saveCandidateProfile(e) {
  e.preventDefault();
  const updatedProfile = {
    id: state.profile ? state.profile.id : 'default',
    name: document.getElementById('profName').value.trim() || 'Candidate',
    target_role: document.getElementById('profTargetRole').value.trim() || 'AI Engineer Intern',
    background: document.getElementById('profBackground').value.trim(),
    skills: document.getElementById('profSkills').value.trim(),
    projects: document.getElementById('profProjects').value.trim(),
    github: document.getElementById('profGithub').value.trim(),
    linkedin: document.getElementById('profLinkedin').value.trim(),
    contact_email: document.getElementById('profEmail').value.trim(),
    tone: document.getElementById('profTone').value,
  };

  try {
    const res = await fetch('/api/profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updatedProfile),
    });
    if (!res.ok) throw new Error('Save profile failed');
    state.profile = await res.json();
    updateProfilePillUI();
    closeProfileModal();
    showToast('Candidate profile updated successfully!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// SETTINGS & LLM CONFIG
// ---------------------------------------------------------------------------

function openSettingsModal() {
  const provider = state.settings.provider || 'gemini';
  if (provider === 'gemini') {
    elements.providerGeminiCard.classList.add('active');
    elements.providerGroqCard.classList.remove('active');
    elements.providerGeminiCard.querySelector('input').checked = true;
  } else {
    elements.providerGroqCard.classList.add('active');
    elements.providerGeminiCard.classList.remove('active');
    elements.providerGroqCard.querySelector('input').checked = true;
  }
  populateModelDropdown(provider);
  elements.settingApiKey.value = state.settings.apiKey || '';
  elements.testResultMsg.classList.add('hidden');
  elements.settingsModal.classList.remove('hidden');
}

function closeSettingsModal() {
  elements.settingsModal.classList.add('hidden');
}

function populateModelDropdown(provider) {
  elements.settingModel.innerHTML = '';
  if (provider === 'gemini') {
    const opt1 = new Option('gemini-3.8-flash (Recommended - High Quality & Fast)', 'gemini-3.8-flash');
    const opt2 = new Option('gemini-3.5-flash', 'gemini-3.5-flash');
    elements.settingModel.add(opt1);
    elements.settingModel.add(opt2);
  } else {
    const opt1 = new Option('llama-3.3-70b-versatile (Groq Free Tier)', 'llama-3.3-70b-versatile');
    const opt2 = new Option('llama-3.1-8b-instant (Groq Ultra-Fast)', 'llama-3.1-8b-instant');
    elements.settingModel.add(opt1);
    elements.settingModel.add(opt2);
  }
  if (state.settings.model) {
    elements.settingModel.value = state.settings.model;
  }
}

function saveSettingsPreferences() {
  const selectedProvider = document.querySelector('input[name="llmProvider"]:checked').value;
  state.settings.provider = selectedProvider;
  state.settings.model = elements.settingModel.value;
  state.settings.apiKey = elements.settingApiKey.value.trim();
  saveSettingsToStorage();
  closeSettingsModal();
  showToast('Settings saved successfully!', 'success');
}

async function testAPIConnection() {
  const provider = document.querySelector('input[name="llmProvider"]:checked').value;
  const model = elements.settingModel.value;
  const apiKey = elements.settingApiKey.value.trim() || null;

  elements.testResultMsg.classList.remove('hidden', 'success', 'error');
  elements.testResultMsg.textContent = 'Testing connection...';

  try {
    const res = await fetch('/api/settings/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: provider,
        model: model,
        api_key: apiKey,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Connection test failed');

    elements.testResultMsg.classList.add('success');
    elements.testResultMsg.textContent = `✓ ${data.message} Model: ${data.model}`;
  } catch (e) {
    elements.testResultMsg.classList.add('error');
    elements.testResultMsg.textContent = `✗ Connection failed: ${e.message}`;
  }
}

// ---------------------------------------------------------------------------
// PIPELINE EXECUTION (SINGLE STREAM)
// ---------------------------------------------------------------------------

async function runSinglePipeline(startupName) {
  if (state.isGenerating) return;
  state.isGenerating = true;

  // Reset & Show Tracker Card
  elements.pipelineTrackerCard.classList.remove('hidden');
  elements.outreachStudioCard.classList.add('hidden');
  resetTrackerUI(startupName);

  const forceRefresh = elements.forceRefreshCheckbox.checked;

  try {
    const res = await fetch('/api/generate-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        startup_name: startupName,
        profile: state.profile,
        llm_config: {
          provider: state.settings.provider,
          model: state.settings.model,
          api_key: state.settings.apiKey || null,
        },
        force_refresh: forceRefresh,
      }),
    });

    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // keep partial

      for (const block of lines) {
        if (!block.startsWith('data: ')) continue;
        const jsonStr = block.replace(/^data:\s*/, '').trim();
        if (!jsonStr) continue;

        try {
          const evt = JSON.parse(jsonStr);
          handlePipelineEvent(evt);
        } catch (err) {
          console.error('Error parsing SSE event:', err);
        }
      }
    }
  } catch (err) {
    appendLog(`[ERROR] ${err.message}`);
    elements.trackerTitle.textContent = 'Pipeline Failed';
    elements.trackerSubtitle.textContent = err.message;
    showToast(`Pipeline failed: ${err.message}`, 'error');
  } finally {
    state.isGenerating = false;
    await refreshSystemStats();
    await loadArchiveApplications();
  }
}

function resetTrackerUI(startupName) {
  elements.trackerTitle.textContent = `Researching & Assessing ${startupName}`;
  elements.trackerSubtitle.textContent = 'Initializing multi-agent pipeline...';
  elements.trackerProgressNum.textContent = '0%';
  elements.progressBarFill.style.width = '0%';
  elements.stepNodes.forEach(node => node.classList.remove('active', 'completed'));
  elements.logsFeed.innerHTML = '';
  appendLog(`[INIT] Target Startup: ${startupName}`);
}

function handlePipelineEvent(evt) {
  if (evt.event === 'step') {
    elements.trackerSubtitle.textContent = evt.message;
    elements.trackerProgressNum.textContent = `${evt.progress}%`;
    elements.progressBarFill.style.width = `${evt.progress}%`;

    // Highlight current and past steps
    const stepNum = evt.step_num || 1;
    elements.stepNodes.forEach((node, idx) => {
      const nodeNum = idx + 1;
      if (nodeNum < stepNum) {
        node.classList.remove('active');
        node.classList.add('completed');
      } else if (nodeNum === stepNum) {
        node.classList.add('active');
        node.classList.remove('completed');
      } else {
        node.classList.remove('active', 'completed');
      }
    });

    appendLog(`[STEP ${stepNum}] ${evt.title}: ${evt.message}`);
  } else if (evt.event === 'log') {
    appendLog(evt.message);
  } else if (evt.event === 'complete') {
    elements.trackerProgressNum.textContent = '100%';
    elements.progressBarFill.style.width = '100%';
    elements.stepNodes.forEach(node => node.classList.add('completed'));
    elements.trackerTitle.textContent = 'Pipeline Complete';
    elements.trackerSubtitle.textContent = evt.message;
    appendLog(`[DONE] ${evt.message}`);

    // Render in Outreach Studio
    renderOutreachStudio(evt.data, evt.slug);
    showToast(`Outreach package ready for ${evt.data.startup}!`, 'success');
  } else if (evt.event === 'error') {
    appendLog(`[ERROR] ${evt.message}`);
    elements.trackerTitle.textContent = 'Pipeline Error';
    elements.trackerSubtitle.textContent = evt.message;
    showToast(evt.message, 'error');
  }
}

function appendLog(text) {
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = text;
  elements.logsFeed.appendChild(line);
  elements.logsFeed.scrollTop = elements.logsFeed.scrollHeight;
}

// ---------------------------------------------------------------------------
// OUTREACH STUDIO RENDERING
// ---------------------------------------------------------------------------

function renderOutreachStudio(data, slug) {
  state.currentApplicationSlug = slug;
  elements.outreachStudioCard.classList.remove('hidden');

  // Header meta
  const name = data.startup || 'Startup';
  elements.studioStartupInitial.textContent = name.charAt(0).toUpperCase();
  elements.studioStartupName.textContent = name;
  elements.studioDateBadge.textContent = data.researched_at || new Date().toISOString().split('T')[0];
  elements.studioSourceBadge.textContent = data.source === 'cache' ? '⚡ From Memory Cache' : '🌐 Fresh Web Research';
  elements.studioTargetRoleDisplay.textContent = `Outreach generated for ${state.profile ? state.profile.target_role : 'Engineering'}`;

  // Fit score dial
  const score = data.fit_score || 0;
  elements.studioFitScore.textContent = score;

  if (score >= 7) {
    elements.studioFitScore.style.color = 'var(--color-success)';
    elements.studioFitLabel.textContent = 'Strong Match';
    elements.studioFitLabel.style.color = 'var(--color-success)';
    elements.studioFitHint.textContent = 'High alignment with your skills. Priority outreach recommended!';
  } else if (score >= 5) {
    elements.studioFitScore.style.color = 'var(--color-warning)';
    elements.studioFitLabel.textContent = 'Moderate Match';
    elements.studioFitLabel.style.color = 'var(--color-warning)';
    elements.studioFitHint.textContent = 'Good potential fit. Consider tailored project emphasis.';
  } else {
    elements.studioFitScore.style.color = 'var(--color-danger)';
    elements.studioFitLabel.textContent = 'Low Match / Skipped';
    elements.studioFitLabel.style.color = 'var(--color-danger)';
    elements.studioFitHint.textContent = 'Fit score below application threshold (5/10).';
  }

  // Company Intel
  elements.studioCompanySummary.textContent = data.company_summary || 'No summary available.';
  renderTechChips(data.tech_stack);

  // Fit Breakdown
  elements.studioFitReasoning.textContent = data.fit_assessment || 'No reasoning available.';
  renderMatchingSkills(data.matching_skills || []);
  renderSkillGaps(data.skill_gaps || []);

  // Outreach Drafts
  elements.studioEmailSubject.value = data.email_subject || `Exploring Opportunities at ${name}`;
  elements.studioColdEmail.value = data.cold_email || '';
  elements.studioLinkedinNote.value = data.linkedin_note || '';
  updateLinkedinCharCount();

  // Setup Mailto link
  updateMailtoLink(name, elements.studioEmailSubject.value, elements.studioColdEmail.value);
}

function renderTechChips(techStackText) {
  elements.studioTechChips.innerHTML = '';
  if (!techStackText || techStackText === 'Unknown') {
    elements.studioTechChips.innerHTML = '<span class="tech-chip">Not publicly disclosed</span>';
    return;
  }
  // Extract bullet points or clean string
  const items = techStackText
    .split(/[\n*•,]+/)
    .map(s => s.replace(/^[-*•\s]+/, '').trim())
    .filter(s => s.length > 1 && !s.toLowerCase().includes('extracted tech stack'));

  if (items.length === 0) {
    elements.studioTechChips.innerHTML = `<span class="tech-chip">${techStackText.slice(0, 50)}</span>`;
    return;
  }

  items.slice(0, 10).forEach(item => {
    const chip = document.createElement('span');
    chip.className = 'tech-chip';
    chip.textContent = item;
    elements.studioTechChips.appendChild(chip);
  });
}

function renderMatchingSkills(skills) {
  elements.studioMatchingSkills.innerHTML = '';
  if (!skills || skills.length === 0) {
    elements.studioMatchingSkills.innerHTML = '<span class="match-chip">Python, Machine Learning</span>';
    return;
  }
  skills.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'match-chip';
    chip.textContent = s;
    elements.studioMatchingSkills.appendChild(chip);
  });
}

function renderSkillGaps(gaps) {
  elements.studioSkillGaps.innerHTML = '';
  if (!gaps || gaps.length === 0) {
    elements.studioSkillGaps.innerHTML = '<span class="gap-chip">None detected</span>';
    return;
  }
  gaps.forEach(g => {
    const chip = document.createElement('span');
    chip.className = 'gap-chip';
    chip.textContent = g;
    elements.studioSkillGaps.appendChild(chip);
  });
}

function updateLinkedinCharCount() {
  const len = elements.studioLinkedinNote.value.length;
  elements.linkedinCharCount.textContent = `${len} / 300`;

  elements.linkedinCharCount.classList.remove('warning', 'danger');
  if (len > 300) {
    elements.linkedinCharCount.classList.add('danger');
  } else if (len > 270) {
    elements.linkedinCharCount.classList.add('warning');
  }
}

function updateMailtoLink(companyName, subject, body) {
  const encSub = encodeURIComponent(subject);
  const encBody = encodeURIComponent(body);
  elements.openMailtoBtn.href = `mailto:?subject=${encSub}&body=${encBody}`;
}

// ---------------------------------------------------------------------------
// BATCH PIPELINE EXECUTION
// ---------------------------------------------------------------------------

async function runBatchPipeline(startupList) {
  if (state.isGenerating) return;
  state.isGenerating = true;

  elements.pipelineTrackerCard.classList.remove('hidden');
  elements.outreachStudioCard.classList.add('hidden');
  resetTrackerUI(`Batch of ${startupList.length} Startups`);

  try {
    const res = await fetch('/api/batch-generate-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        startups: startupList,
        profile: state.profile,
        llm_config: {
          provider: state.settings.provider,
          model: state.settings.model,
          api_key: state.settings.apiKey || null,
        },
        force_refresh: false,
      }),
    });

    if (!res.ok) throw new Error('Batch execution failed');

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const block of lines) {
        if (!block.startsWith('data: ')) continue;
        const jsonStr = block.replace(/^data:\s*/, '').trim();
        if (!jsonStr) continue;

        try {
          const evt = JSON.parse(jsonStr);
          if (evt.event === 'batch_item_start') {
            appendLog(`\n========================================\n[BATCH ${evt.index}/${evt.total}] Starting: ${evt.startup}`);
            elements.trackerTitle.textContent = `Batch Processing: ${evt.startup} (${evt.index}/${evt.total})`;
          } else if (evt.event === 'batch_complete') {
            appendLog(`[BATCH DONE] Successfully processed ${evt.total} startups.`);
            elements.trackerTitle.textContent = 'Batch Sprint Completed!';
            showToast(`Batch completed! Processed ${evt.total} startups.`, 'success');
          } else {
            handlePipelineEvent(evt);
          }
        } catch (e) {
          console.error('Batch parse error', e);
        }
      }
    }
  } catch (err) {
    showToast(`Batch failed: ${err.message}`, 'error');
  } finally {
    state.isGenerating = false;
    await refreshSystemStats();
    await loadArchiveApplications();
  }
}

// ---------------------------------------------------------------------------
// DISCOVERY & VECTOR MEMORY
// ---------------------------------------------------------------------------

async function loadDiscoveryItems(query) {
  elements.discoveryResultsGrid.innerHTML = '<div class="text-muted p-4">Searching ChromaDB vector space...</div>';

  try {
    const res = await fetch(`/api/discovery/similar?query=${encodeURIComponent(query)}&n=8`);
    if (!res.ok) throw new Error('Discovery query failed');
    const data = await res.json();

    elements.discoveryResultsTitle.textContent = `Results for "${query}"`;
    elements.discoveryResultsCount.textContent = `${data.count} matches found in memory`;

    if (data.results.length === 0) {
      elements.discoveryResultsGrid.innerHTML = `
        <div class="col-span-2 card-sub text-center p-6">
          <p class="text-secondary mb-2">No companies matched this semantic query yet.</p>
          <span class="text-muted text-xs">Run the pipeline on new startups to automatically grow your ChromaDB vector memory.</span>
        </div>`;
      return;
    }

    elements.discoveryResultsGrid.innerHTML = '';
    data.results.forEach(item => {
      const card = document.createElement('div');
      card.className = 'discovery-card';

      const simPercent = Math.round((item.similarity_score || 0.5) * 100);
      card.innerHTML = `
        <div class="discovery-card-top">
          <h4 class="discovery-card-name">${item.startup_name}</h4>
          <span class="similarity-badge">Match: ${simPercent}%</span>
        </div>
        <p class="discovery-card-summary">${item.company_research || 'No summary available.'}</p>
        <div class="tech-chips-wrap">
          <span class="tech-chip text-xs">${(item.tech_stack || 'Tech disclosed').slice(0, 40)}</span>
        </div>
        <div class="mt-auto pt-3 border-t border-subtle flex justify-between items-center">
          <span class="text-xs text-muted">In Vector Store</span>
          <button class="btn-sm btn-primary-gradient apply-from-disc-btn" data-company="${item.startup_name}">
            Generate Outreach
          </button>
        </div>
      `;

      card.querySelector('.apply-from-disc-btn').addEventListener('click', () => {
        elements.targetStartupInput.value = item.startup_name;
        switchTab('generator');
        runSinglePipeline(item.startup_name);
      });

      elements.discoveryResultsGrid.appendChild(card);
    });
  } catch (e) {
    elements.discoveryResultsGrid.innerHTML = `<div class="text-danger p-4">Error: ${e.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// ARCHIVE & SAVED APPLICATIONS
// ---------------------------------------------------------------------------

async function loadArchiveApplications() {
  const searchQuery = elements.archiveSearchInput.value.trim();
  let url = `/api/applications?status=${state.archiveFilter}`;
  if (searchQuery) url += `&q=${encodeURIComponent(searchQuery)}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load archive');
    const data = await res.json();
    const apps = data.applications || [];

    // Update Metrics
    updateArchiveMetrics(apps);
    elements.archiveCountBadge.textContent = apps.length;

    if (apps.length === 0) {
      elements.archiveGrid.innerHTML = `
        <div class="col-span-2 card-sub text-center p-8">
          <p class="text-secondary mb-2">No application records found matching current filter.</p>
          <span class="text-muted text-xs">Run the pipeline on startups to populate your local archive.</span>
        </div>`;
      return;
    }

    elements.archiveGrid.innerHTML = '';
    apps.forEach(app => {
      const card = document.createElement('div');
      card.className = 'archive-card';

      let badgeClass = 'badge-med';
      if (app.fit_score >= 7) badgeClass = 'badge-high';
      else if (app.fit_score < 5 || app.error) badgeClass = 'badge-skip';

      card.innerHTML = `
        <div class="archive-card-top">
          <h4 class="archive-startup-name">${app.startup}</h4>
          <span class="archive-fit-badge ${badgeClass}">${app.fit_score}/10 Fit</span>
        </div>
        <p class="archive-card-summary">${app.company_summary || 'No summary available.'}</p>
        <div class="archive-card-footer">
          <span class="text-xs text-muted">${app.researched_at || 'Archived'}</span>
          <div class="flex gap-2">
            <button class="btn-sm btn-outline open-app-btn" data-slug="${app.slug}">Open Studio</button>
            <button class="btn-sm btn-icon delete-app-btn" data-slug="${app.slug}" title="Delete record">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </div>
      `;

      card.querySelector('.open-app-btn').addEventListener('click', async () => {
        try {
          const detailRes = await fetch(`/api/applications/${app.slug}`);
          const detailData = await detailRes.json();
          renderOutreachStudio(detailData, app.slug);
          switchTab('generator');
        } catch (err) {
          showToast('Could not load record', 'error');
        }
      });

      card.querySelector('.delete-app-btn').addEventListener('click', async () => {
        if (!confirm(`Delete application for ${app.startup}?`)) return;
        try {
          const delRes = await fetch(`/api/applications/${app.slug}`, { method: 'DELETE' });
          if (!delRes.ok) throw new Error('Delete failed');
          showToast(`Deleted ${app.startup}`, 'success');
          loadArchiveApplications();
          refreshSystemStats();
        } catch (err) {
          showToast(err.message, 'error');
        }
      });

      elements.archiveGrid.appendChild(card);
    });
  } catch (e) {
    elements.archiveGrid.innerHTML = `<div class="text-danger p-4">Error loading archive: ${e.message}</div>`;
  }
}

function updateArchiveMetrics(apps) {
  elements.metricTotalApps.textContent = apps.length;
  elements.metricHighFit.textContent = apps.filter(a => a.fit_score >= 7 && !a.error).length;
  elements.metricMedFit.textContent = apps.filter(a => a.fit_score >= 5 && a.fit_score < 7 && !a.error).length;
  elements.metricSkipped.textContent = apps.filter(a => a.fit_score < 5 || a.skipped || a.error).length;
}

// ---------------------------------------------------------------------------
// SYSTEM STATS & HELPERS
// ---------------------------------------------------------------------------

async function refreshSystemStats() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) return;
    const health = await res.json();
    const count = health.chroma_memory_count || 18;
    elements.memoryCountBadge.textContent = count;
    elements.discoveryTotalCount.textContent = count;
  } catch (e) {
    console.warn('Health check error', e);
  }
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
