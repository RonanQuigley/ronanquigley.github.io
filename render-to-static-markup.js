import React from 'react';
import fs from 'fs';
import ReactDOM from 'react-dom/server';
import App from './app';
import { ServerStyleSheet } from 'styled-components';
import { minify } from 'html-minifier';

const sheet = new ServerStyleSheet();

try {
    const markup = ReactDOM.renderToStaticMarkup(sheet.collectStyles(<App />));
    const styleTags = sheet.getStyleTags();
    const html = minify(
        `
        <html lang="en-GB">
            <head>             
                <title>Ronan Quigley - Software Engineer</title>
                <link rel="icon" type="image/x-icon" href="./favicon.ico">
                <meta charSet="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <!-- Google tag (gtag.js) -->
                <script async src="https://www.googletagmanager.com/gtag/js?id=G-N5SBRCG7YQ"></script>
                <script>
                  window.dataLayer = window.dataLayer || [];
                  function gtag(){dataLayer.push(arguments);}
                  gtag('js', new Date());
                
                  gtag('config', 'G-N5SBRCG7YQ');
                </script>                
                ${styleTags}
            </head>
            <body>
                <div id="root">
                    ${markup}
                </div>
                <script src="./assets/deobfuscate.js"></script>
            </body>
        </html>
    `,
        {
            collapseWhitespace: true,
            minifyCSS: true,
            removeComments: true,
            html5: true,
            // eslint-disable-next-line quotes
            quoteCharacter: "'",
        },
    );
    fs.writeFileSync('./index.html', html);
} catch (error) {
    console.error(error);
} finally {
    sheet.seal();
}
