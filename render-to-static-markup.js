import React from 'react';
import fs from 'fs';
import ReactDOM from 'react-dom/server';
import App from './src/app';
import { ServerStyleSheet } from 'styled-components';
import { minify } from 'html-minifier';

const sheet = new ServerStyleSheet();

try {
    const markup = ReactDOM.renderToStaticMarkup(sheet.collectStyles(<App />));
    const styleTags = sheet.getStyleTags();
    const html = minify(
        `
        <html>
            <head>
                ${styleTags}
            </head>
            <body>
                <div id="root">
                    ${markup}
                </div>
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
