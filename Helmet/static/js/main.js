// Variables to store the selected file
let selectedFile = null;

// Get DOM elements
const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const gallery = document.getElementById('gallery');
const processBtn = document.getElementById('process-btn');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress');
const progressText = document.getElementById('progress-text');
const errorDisplay = document.getElementById('error');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// Handle selected files
function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        
        // Check if file is a video
        if (!selectedFile.type.startsWith('video/')) {
            showError('Please select a video file');
            return;
        }
        
        // Clear previous gallery items
        gallery.innerHTML = '';
        
        // Display file info
        const fileInfo = document.createElement('div');
        fileInfo.className = 'video-info';
        
        // Create video thumbnail
        const video = document.createElement('video');
        video.className = 'thumb';
        video.width = 300;
        video.src = URL.createObjectURL(selectedFile);
        video.muted = true;
        video.autoplay = false;
        video.controls = true;
        
        // Create file name display
        const fileName = document.createElement('p');
        fileName.className = 'video-name';
        fileName.textContent = selectedFile.name;
        
        // Create file size display
        const fileSize = document.createElement('p');
        fileSize.textContent = `Size: ${formatFileSize(selectedFile.size)}`;
        
        // Add to gallery
        fileInfo.appendChild(video);
        fileInfo.appendChild(fileName);
        fileInfo.appendChild(fileSize);
        gallery.appendChild(fileInfo);
        
        // Enable process button
        processBtn.classList.remove('disabled');
        processBtn.disabled = false;
        
        // Clear any previous errors
        hideError();
    }
}

// Process button click handler
processBtn.addEventListener('click', uploadAndProcess);

function uploadAndProcess() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    // Create FormData object
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Show progress container
    progressContainer.classList.remove('hidden');
    
    // Disable process button during upload
    processBtn.classList.add('disabled');
    processBtn.disabled = true;
    
    // Upload file with progress tracking
    const xhr = new XMLHttpRequest();
    
    xhr.open('POST', '/upload', true);
    
    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percentComplete = Math.round((e.loaded / e.total) * 100);
            progressBar.style.width = percentComplete + '%';
            progressText.textContent = percentComplete + '%';
        }
    };
    
    xhr.onload = function() {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            
            if (response.success) {
                // Redirect to results page
                window.location.href = `/result/${response.task_id}`;
            } else {
                showError(response.error || 'Upload failed');
                resetUploadUI();
            }
        } else {
            showError('Upload failed: ' + xhr.statusText);
            resetUploadUI();
        }
    };
    
    xhr.onerror = function() {
        showError('Network error occurred');
        resetUploadUI();
    };
    
    // Send the form data
    xhr.send(formData);
}

// Reset the UI after upload completes or fails
function resetUploadUI() {
    processBtn.classList.remove('disabled');
    processBtn.disabled = false;
    progressContainer.classList.add('hidden');
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
}

// Show error message
function showError(message) {
    errorDisplay.classList.remove('hidden');
    errorDisplay.innerHTML = `<p>${message}</p>`;
}

// Hide error message
function hideError() {
    errorDisplay.classList.add('hidden');
    errorDisplay.innerHTML = '';
}

// Format file size to human-readable format
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}