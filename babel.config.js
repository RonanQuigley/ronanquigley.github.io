module.exports = (api) => {
    // This caches the Babel config
    api.cache(true);
    return {
        presets: ['@babel/preset-env', '@babel/preset-react'],
        plugins:
            process.env.NODE_ENV !== 'BUILD'
                ? [
                      'react-refresh/babel',
                      [
                          'babel-plugin-styled-components',
                          {
                              pure: true,
                          },
                      ],
                  ]
                : [
                      [
                          'babel-plugin-styled-components',
                          {
                              pure: true,
                          },
                      ],
                  ],
    };
};
