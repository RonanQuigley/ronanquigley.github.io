(function () {
    setTimeout(() => {
        try {
            const email = 'bWFpbHRvOmluZm9Acm9uYW5xdWlnbGV5LmNvbQ==';
            const emailElement = document.getElementById('email');
            const decodedEmail = window.atob(email);
            emailElement.href = decodedEmail;
        } catch (error) {
            console.error('Could not setup email href: ', error);
        }
    }, 1000);
})();
