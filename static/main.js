require.config({
    paths: {
        'jquery': 'https://code.jquery.com/jquery-3.6.0.min',
        'popper': 'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.1/umd/popper.min',
        'bootstrap': 'https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min'
    },
    shim: {
        'bootstrap': {
            deps: ['jquery', 'popper']
        }
    }
});

require(['bootstrap'], function() {
});