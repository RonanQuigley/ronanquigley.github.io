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
        padding: 0;
        @font-face {
          font-family: 'Indie Flower';
          font-style: normal;
          font-weight: 400;
          src: url(https://fonts.gstatic.com/s/indieflower/v16/m8JVjfNVeKWVnh3QMuKkFcZVaUuH99GUDg.woff2) format('woff2');
          unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
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
        font-size: 1.4rem;
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
    width: 90%;
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 0px;
    @media (min-width: 1200px) {
        padding: 50px 0px 0px;
    }
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
`;

const CardImage = styled.img`
    object-fit: contain;
    ${borderCssMixIn}
`;

const FmpCardImage = styled(CardImage)`
    width: 70%;
    @media (min-width: 720px) {
        min-width: 300px;
        width: 40%;
    }
    @media (min-width: 1200px) {
        min-width: unset;
        width: 57%;
    }
`;

const SbCardImage = styled(CardImage)`
    width: 100%;
`;

const CardInnerContainer = styled.div`
    padding: 0px 8px 0px 8px;
    width: 100%;
`;

const CardInnerContainerImage = styled(CardInnerContainer)`
    text-align: center;
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
`;

const LatestWorkTextContainer = styled.div`
    width: 150px;
    margin: 40px auto;
`;

const App = () => (
    <>
        <GlobalStyle />
        <OuterContainer>
            <HeaderSection>
                <MainHeading />
            </HeaderSection>
            <NavbarContainer>
                <BlackButton href="./assets/cv.pdf">CV</BlackButton>
                {/** do not add the href tag to prevent spam */}
                <BlackButton id="email">Email</BlackButton>
                <BlackButton
                    target="_blank"
                    rel="noopener noreferrer"
                    href="https://github.com/ronanquigley"
                >
                    Github
                </BlackButton>
                <BlackButton href="/blog">Blog</BlackButton>
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
                            <WhiteButton
                                target="_blank"
                                rel="noopener noreferrer"
                                href="https://findmypast.co.uk/"
                            >
                                Take a look
                            </WhiteButton>
                        </CardInnerContainer>
                        <CardInnerContainerImage>
                            <FmpCardImage
                                alt="The company logo for find my past."
                                src="./assets/fmp.jpg"
                            />
                        </CardInnerContainerImage>
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
                            <WhiteButton
                                target="_blank"
                                rel="noopener noreferrer"
                                href="https://store.steampowered.com/app/290060/Glitchspace/"
                            >
                                Take a look
                            </WhiteButton>
                        </CardInnerContainer>
                        <CardInnerContainerImage>
                            <SbCardImage
                                alt="A screenshot taken from the PC game Glitchspace, released on Steam in 2016."
                                src="./assets/glitchspace.jpg"
                            />
                        </CardInnerContainerImage>
                    </CardContent>
                </CardOuterContainer>
            </MainContent>
        </OuterContainer>
    </>
);

export default App;
