import React from 'react';
import styled, { css } from 'styled-components';
import { createGlobalStyle } from 'styled-components';
import MainHeading from './main-heading';
import LatestWorkText from './latest-work-text';

const COLOURS = {
    RICH_BLACK: '#080708',
    WHITE: '#FBFBFB',
    INDIGO: '#323456',
    BLUE: '#568EC5',
};

const borderCssMixIn = css`
    border-radius: 8px;
    box-shadow: rgb(45 43 36 / 20%) 0px 0px 20px 0px;
`;

const hoverEmphasis = css`
    box-shadow: rgb(45 43 36 / 80%) 0px 0px 10px 0px;
    filter: brightness(150%);
`;

const CardOuterContainer = ({ colour, children }) => (
    <article
        css={css`
            background-color: ${colour};
            width: 100%;
            ${borderCssMixIn}
        `}
    >
        {children}
    </article>
);

const GlobalStyle = createGlobalStyle`
    body {
        background-color: ${COLOURS.BACKGROUND}; 
        margin: 0 auto;   
        @media (min-width: 1200px) {            
            padding: 100px;
        }   
    }
    body, input, textarea, button  {
        font-family: "Indie Flower", sans-serif;
        font-size: 62.5%;
        font-weight: normal;
    }
    h2 {
        font-size: 1.6rem;
    }
    p, a {
        font-size: 1.2rem;
    }
`;

const Button = styled.a`
    ${borderCssMixIn}
    border: none;
    width: 130px;
    height: 36px;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    &:hover {
        ${hoverEmphasis}
    }
`;

const BlackButton = styled(Button)`
    background-color: ${COLOURS.RICH_BLACK};
    color: ${COLOURS.WHITE};
`;

const WhiteButton = styled(Button)`
    background-color: ${COLOURS.WHITE};
    color: ${COLOURS.RICH_BLACK};
`;

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

const NavbarContainer = styled.nav`
    display: flex;
    width: 100%;
    max-width: 600px;
    flex-direction: column;
    justify-content: space-between;
    height: 180px;
    align-items: center;
    margin: 10px auto 0px;
    @media (min-width: 720px) {
        margin: 8px auto 24px;
        justify-content: space-evenly;
        flex-direction: row;
        height: auto;
    }
`;

const OuterContainer = styled.div`
    width: 95%;
    max-width: 1280px;
    margin: 0 auto;
`;

const MainContent = styled.main`
    display: grid;
    grid-gap: 40px;
    margin: 0 auto;
    @media (min-width: 1200px) {
        grid-template-columns: 1fr 1fr;
    }
`;

const CardContent = styled.div`
    display: flex;
    color: ${COLOURS.WHITE};
    padding: 30px;
    margin: 0 auto;
    width: auto;
    height: auto;
    flex-direction: column-reverse;
    @media (min-width: 720px) {
        flex-direction: row;
    }
    @media (min-width: 1200px) {
        height: 500px;
    }
`;

const CardImage = styled.img`
    object-fit: cover;
    width: 100%;
    ${borderCssMixIn}
`;

const CardInnerContainer = styled.div`
    padding: 0px 8px 0px 8px;
    width: 100%;
    @media (min-width: 720px) {
        width: 50%;
    }
`;

const CardHeadingText = styled.div`
    text-decoration: underline;
`;

const CardText = styled.div`
    margin: 40px 0;
`;

const HeaderSection = styled.header`
    margin: 0 auto;
    max-width: 400px;
    padding: 24px 0;
    @media (min-width: 1200px) {
        padding: 0;
    }
`;

const LatestWorkTextContainer = styled.div`
    width: 150px;
    margin: 40px auto;
`;

const App = () => (
    <>
        <link
            href="https://fonts.googleapis.com/css?family=Indie+Flower"
            rel="stylesheet"
        />
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <GoogleTagManager />
        <GlobalStyle />
        <OuterContainer>
            <HeaderSection>
                <MainHeading />
            </HeaderSection>
            <NavbarContainer>
                <BlackButton href="./assets/cv.pdf">CV</BlackButton>
                <BlackButton href="mailto:info@ronanquigley.com">
                    Email
                </BlackButton>
                <BlackButton href="https://github.com/ronanquigley">
                    Github
                </BlackButton>
            </NavbarContainer>
            <LatestWorkTextContainer>
                <LatestWorkText />
            </LatestWorkTextContainer>
            <MainContent>
                <CardOuterContainer colour={COLOURS.INDIGO}>
                    <CardContent>
                        <CardInnerContainer>
                            <CardHeadingText as="h2">
                                Findmypast
                            </CardHeadingText>
                            <CardText as="p">
                                I&apos;m currently working as a senior software
                                engineer on a family history product featuring
                                billions of searchable records. I&apos;ve worked
                                across cross-functional and self-directed teams,
                                in both full stack and devops capacities.
                            </CardText>
                            <WhiteButton href="https://findmypast.co.uk/">
                                Take a look
                            </WhiteButton>
                        </CardInnerContainer>
                        <CardInnerContainer>
                            <CardImage
                                alt="The company logo for find my past."
                                src="./assets/fmp.jpg"
                            />
                        </CardInnerContainer>
                    </CardContent>
                </CardOuterContainer>
                <CardOuterContainer colour={COLOURS.BLUE}>
                    <CardContent>
                        <CardInnerContainer>
                            <CardHeadingText as="h2">
                                Space Budgie
                            </CardHeadingText>
                            <CardText as="p">
                                I co-founded an independent games studio. Our
                                flagship title Glitchspace, a first-person
                                visual programming game, was developed over the
                                course of three years. The game won multiple
                                awards, including a Scottish BAFTA.
                            </CardText>
                            <WhiteButton href="https://store.steampowered.com/app/290060/Glitchspace/">
                                Take a look
                            </WhiteButton>
                        </CardInnerContainer>
                        <CardInnerContainer>
                            <CardImage
                                alt="screenshot taken from the PC game Glitchspace, released on Steam in 2016."
                                src="./assets/glitchspace.jpg"
                            />
                        </CardInnerContainer>
                    </CardContent>
                </CardOuterContainer>
            </MainContent>
        </OuterContainer>
    </>
);

export default App;
