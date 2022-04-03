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

const Card = ({ colour, children }) => (
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
        padding: 100px;   
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
    max-width: 1280px;
    margin: 0 auto;
`;

const MainContent = styled.main`
    display: grid;
    grid-gap: 40px;
    @media (min-width: 1200px) {
        grid-template-columns: 1fr 1fr;
    }
`;

const CardContent = styled.div`
    display: flex;
    flex-direction: row;
    color: ${COLOURS.WHITE};
    width: auto;
    padding: 30px;
    height: 500px;
    margin: 0 auto;
`;

const CardImage = styled.img`
    object-fit: cover;
    width: 100%;
    ${borderCssMixIn}
`;

const CardLeft = styled.div`
    width: 50%;
    padding: 0px 8px 0px 8px;
`;

const CardRight = styled.div`
    width: 50%;
    padding: 0px 8px 0px 8px;
`;

const CardHeadingText = styled.div`
    text-decoration: underline;
`;

const CardText = styled.div`
    margin: 40px 0;
`;

const HeaderSection = styled.header`
    margin: 0 auto;
    width: 400px;
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
                <Card colour={COLOURS.INDIGO}>
                    <CardContent>
                        <CardLeft>
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
                        </CardLeft>
                        <CardRight>
                            <CardImage
                                alt="The company logo for find my past."
                                src="./assets/fmp.jpg"
                            />
                        </CardRight>
                    </CardContent>
                </Card>
                <Card colour={COLOURS.BLUE}>
                    <CardContent>
                        <CardLeft>
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
                        </CardLeft>
                        <CardRight>
                            <CardImage
                                alt="screenshot taken from the PC game Glitchspace, released on Steam in 2016."
                                src="./assets/glitchspace.jpg"
                            />
                        </CardRight>
                    </CardContent>
                </Card>
            </MainContent>
        </OuterContainer>
    </>
);

export default App;
