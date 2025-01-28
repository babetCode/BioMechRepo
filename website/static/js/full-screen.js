document.querySelectorAll('.fullscreenBtn').forEach(button => {
    button.addEventListener('click', function() {
        var iframe = this.parentElement.querySelector('iframe');
        if (iframe && iframe.tagName === 'IFRAME') {
            if (iframe.requestFullscreen) {
                iframe.requestFullscreen();
            } else if (iframe.mozRequestFullScreen) { // Firefox
                iframe.mozRequestFullScreen();
            } else if (iframe.webkitRequestFullscreen) { // Chrome, Safari, and Opera
                iframe.webkitRequestFullscreen();
            } else if (iframe.msRequestFullscreen) { // IE/Edge
                iframe.msRequestFullscreen();
            }
        } else {
            console.error('No iframe found next to the button.');
        }
    });
});