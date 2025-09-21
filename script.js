// JavaScript for the Login Page
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevents page refresh
        // In a real app, you would have authentication logic here
        window.location.href = "/dashboard"; // Redirects to the dashboard
    });
}


// JavaScript for the OMR Grader Page (index.html)
document.addEventListener('DOMContentLoaded', function() {
    const excelInput = document.getElementById('file-upload-1');
    const folderInput = document.getElementById('file-upload-2');
    const excelLabel = document.getElementById('answer-filename');
    const folderLabel = document.getElementById('folder-selection-text');

    // Updates the label text for the answer key file input
    if (excelInput && excelLabel) {
        excelInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                excelLabel.textContent = this.files[0].name;
            } else {
                excelLabel.textContent = "Choose CSV file";
            }
        });
    }

    // Updates the label text for the OMR folder input
    if (folderInput && folderLabel) {
        folderInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                folderLabel.textContent = `${this.files.length} file(s) selected`;
            } else {
                folderLabel.textContent = "Select OMR folder";
            }
        });
    }
});

