import React from 'react';
import styled, { css } from 'styled-components';
import { createGlobalStyle } from 'styled-components';
import CVIcon from './cv-icon';
import GithubIcon from './github-icon';
import EmailIcon from './email-icon';

const COLOURS = {
    BACKGROUND: '#FFFFFA',
    PRIMARY: '#FFF8F0',
    SECONDARY: '#1B1B1E',
    FONT: '#0C0C0C',
    FONT_SECONDARY: '#FFFFFC',
};

const SHADOWS = {
    HIGH: 'box-shadow: rgb(88 43 36 / 30%) 0px 0px 20px 0px;',
    MEDIUM: 'box-shadow: rgb(66 43 36 / 20%) 0px 0px 20px 0px;',
    LIGHT: 'box-shadow: rgb(45 43 36 / 10%) 0px 0px 20px 0px;',
};

const GlobalStyle = createGlobalStyle`

    body {
        background-color: ${COLOURS.BACKGROUND};
        font-family: 'Roboto', sans-serif;;
    }
`;

const Image = styled.img`
    object-fit: cover;
    width: ${(props) => props.width ?? '400px'};
    height: ${(props) => props.height ?? '300px'};
    border-radius: 8px;
    ${SHADOWS.LIGHT}
    color: magenta;
`;

const Article = styled.article`
    display: flex;
    flex-direction: column;
    background-color: ${COLOURS.SECONDARY};
    margin: 32px;
    border-radius: 8px;
    ${SHADOWS.MEDIUM}
    &:hover {
        cursor: pointer;
        transform: scale(1.02);
    }
`;

const Text = styled.div`
    text-align: center;
`;

const ImageWithLink = ({ href, alt, imgAssetPath, width, height }) => (
    <WithLink href={href}>
        <Image alt={alt} src={imgAssetPath} width={width} height={height} />
    </WithLink>
);

const WorkContainer = styled.div`
    display: block;
    background-color: ${COLOURS.PRIMARY};
    border-radius: 16px;
    ${SHADOWS.HIGH}
    @media (min-width: 600px) {
        display: grid;
        grid-template-columns: repeat(auto-fit, 1fr);
    }
`;

const WithLink = ({ href, children }) => (
    <a
        href={href}
        css={css`
            margin: 0 auto;
            padding: 8px;
        `}
    >
        {children}
    </a>
);

const ArticleText = ({ children }) => (
    <Text
        as="p"
        css={css`
            width: 80%;
            margin: 16px auto;
            color: ${COLOURS.FONT_SECONDARY};
            text-align: center;
        `}
    >
        {children}
    </Text>
);

const GoogleTagManager = () => (
    <>
        <script
            async
            src="https://www.googletagmanager.com/gtag/js?id=UA-82512705-1"
        ></script>
        <script
            dangerouslySetInnerHTML={{
                __html: `window.dataLayer = window.dataLayer || [];
                        function gtag() { dataLayer.push(arguments); }
                        gtag('js', new Date());
                        gtag('config', 'UA-82512705-1');
                    `,
            }}
        />
    </>
);

const App = () => (
    <>
        <GoogleTagManager />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin />
        <link
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@300&display=swap"
            rel="stylesheet"
        />
        <GlobalStyle />
        <div
            css={css`
                margin: 0 auto;
                max-width: 1280px;
            `}
        >
            <header
                css={`
                    grid-column: 1 / span 2;
                    margin: 40px;
                    /* background-color: ${COLOURS.PRIMARY}; */
                `}
            >
                <Text
                    as="h1"
                    css={css`
                        color: ${COLOURS.FONT_PRIMARY};
                    `}
                >
                    Ronan Quigley
                </Text>
                <Text
                    as="h3"
                    css={css`
                        color: ${COLOURS.FONT_PRIMARY};
                    `}
                >
                    Full Stack Developer
                </Text>
                <div
                    css={css`
                        display: flex;
                        flex-direction: column;
                        width: 100%;
                        align-items: center;
                        @media (min-width: 600px) {
                            flex-direction: row;
                            width: 30%;
                            min-width: 400px;
                            margin: 0 auto;
                            justify-content: space-between;
                        }
                    `}
                >
                    <WithLink>
                        <GithubIcon
                            width="100px"
                            height="100px"
                            fillColour={COLOURS.FONT_PRIMARY}
                        />
                    </WithLink>
                    <WithLink>
                        <EmailIcon
                            width="100px"
                            height="100px"
                            fillColour={COLOURS.FONT_PRIMARY}
                        />
                    </WithLink>
                    <WithLink>
                        <CVIcon
                            width="100px"
                            height="100px"
                            fillColour={COLOURS.FONT_PRIMARY}
                        />
                    </WithLink>
                </div>
            </header>
            <WorkContainer>
                <Text
                    as="h3"
                    css={css`
                        color: ${COLOURS.FONT};
                        grid-column: 1 / span 2;
                        text-decoration: underline;
                        margin-bottom: 0;
                    `}
                >
                    Latest Work
                </Text>
                <Article>
                    <Text
                        as="h4"
                        css={css`
                            text-decoration: underline;
                            color: ${COLOURS.FONT_SECONDARY};
                        `}
                    >
                        Findmypast
                    </Text>
                    <ImageWithLink
                        imgAssetPath="./assets/fmp.jpg"
                        href="https://findmypast.co.uk"
                        alt="Findmypast branding; external link that to the Findmypast website"
                    />
                    <ArticleText as="p">
                        I'm currently working as a full-stack software engineer
                        at Findmypast, a family history site that provides
                        worldwide users with access to billions of genealogy
                        records.
                    </ArticleText>
                </Article>
                <Article>
                    <Text
                        as="h4"
                        css={css`
                            text-decoration: underline;
                            color: ${COLOURS.FONT_SECONDARY};
                        `}
                    >
                        Glitchspace
                    </Text>
                    <ImageWithLink
                        imgAssetPath="./assets/glitchspace.jpg"
                        href="https://store.steampowered.com/app/290060/Glitchspace/"
                        alt="In game screenshot of Glitchspace; external link that to the Steam store page"
                    />
                    <ArticleText as="p">
                        A first-person visual programming game. Developed at
                        Space Budgie from a previous career in games
                        development. Winner of multiple awards, including a
                        Scottish BAFTA.
                    </ArticleText>
                </Article>
            </WorkContainer>
        </div>
    </>
);

export default App;
