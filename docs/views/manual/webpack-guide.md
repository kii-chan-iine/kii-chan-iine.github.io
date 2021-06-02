---
author: 星火燎原@vxhly
title: webpack 初学指南
categories: [manual]
tags: [Node.js]
date: 2018-11-16 14:25:54
---

<Boxx changeTime="10000"/>

::: tip 前言
这边用到的 webpack 版本 4.0+, 这边将接触到 TypeScript, ES6, ES7, sass 等 loader ... 不想解释了直接上代码。
:::
<!-- more -->

## 目录结构

``` bash
--- dist
  |- css
  |- js
  |- images
  |- fonts
  |- index.html
--- src
  |- scss
  |- js
  |- images
  |- fonts
  |- index.html
```

## package.json

``` json
{
  "name": "webpack-demo",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "serve": "webpack-dev-server --compress --progress ",
    "dev": "webpack --mode development",
    "build": "webpack --mode production"
  },
  "author": "",
  "license": "MIT",
  "devDependencies": {
    "@babel/core": "^7.1.5",
    "@babel/preset-env": "^7.1.5",
    "autoprefixer": "^9.3.1",
    "babel-cli": "^6.26.0",
    "babel-eslint": "^10.0.1",
    "babel-loader": "^8.0.4",
    "babel-preset-env": "^1.7.0",
    "clean-webpack-plugin": "^1.0.0",
    "compression-webpack-plugin": "^2.0.0",
    "css-loader": "^1.0.1",
    "eslint": "^5.9.0",
    "eslint-loader": "^2.1.1",
    "file-loader": "^2.0.0",
    "html-webpack-plugin": "^3.2.0",
    "html-withimg-loader": "^0.1.16",
    "mini-css-extract-plugin": "^0.4.4",
    "node-sass": "^4.10.0",
    "open-browser-webpack-plugin": "0.0.5",
    "optimize-css-assets-webpack-plugin": "^5.0.1",
    "postcss-loader": "^3.0.0",
    "sass-loader": "^7.1.0",
    "standard": "^12.0.1",
    "style-loader": "^0.23.1",
    "ts-loader": "^5.3.0",
    "typescript": "^3.1.6",
    "url-loader": "^1.1.2",
    "webpack": "^4.25.1",
    "webpack-cli": "^3.1.2",
    "webpack-dev-server": "^3.1.10"
  }
}
```

## .babelrc

``` json
{
  "presets": ["@babel/preset-env"]
}
```

## .eslintrc.js

``` javascript
module.exports = {
    root: true,
    env: {
        node: true,
        browser: true
    },
    extends: [
        'standard'
    ]
}
```

## tsconfig.json

``` json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "es5",
    "sourceMap": true,
    "experimentalDecorators": true
  },
  "exclude": [
    "node_modules"
  ]
}
```

## webpack.config.js

``` javascript
const path = require('path')
// 拆分 css 出来单独打包
const MiniCssExtractPlugin = require('mini-css-extract-plugin')
// 添加不同浏览器的兼容
const autoprefixer = require('autoprefixer')
// 打包之前清掉原打包信息
const CleanWebpackPlugin = require('clean-webpack-plugin')
// 结合 postcss 来解决不同浏览器的兼容
const OptimizeCssAssetsPlugin = require('optimize-css-assets-webpack-plugin')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const OpenBrowserPlugin = require('open-browser-webpack-plugin')
const CompressionWebpackPlugin = require('compression-webpack-plugin')

const productionGzipExtensions = ['js', 'css']

module.exports = {
        mode: 'development',
        entry: path.join(__dirname, './src/js/main.ts'),
        output: {
            filename: 'js/[name]-[hash:32].js',
            path: path.resolve(__dirname, 'dist')
        },
        resolve: {
            extensions: ['.js', '.json', '.css', '.scss', '.sass', 'ts'],
            alias: {
                '@': path.join(__dirname, 'src')
            }
        },
        module: {
            rules: [

                {
                    test: /\.ts$/,
                    exclude: /node_modules/,
                    use: 'ts-loader'
                },
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: [

                        {
                            loader: 'babel-loader'
                        }

                    ]

                },
                {
                    test: /\.(css)$/,
                    // exclude: /node_modules/,
                    use: [{
                        loader: 'css-loader',
                        options: {
                            importLoaders: 2
                        }
                    }]
                },
                {
                    test: /\.(sass|scss)$/,
                    // exclude: /node_modules/,
                    use: [

                        {
                            loader: MiniCssExtractPlugin.loader
                        },
                        {
                            loader: 'css-loader',
                            options: {
                                importLoaders: 2
                            }
                        },
                        {
                            loader: 'postcss-loader',
                            options: {
                                plugins: () => [
                                    autoprefixer({
                                        browsers: [
                                            'ie >= 8',
                                            'ie_mob >= 10',
                                            'ff >= 26',
                                            'chrome >= 20',
                                            'safari >= 6',
                                            'opera >= 12',
                                            'Firefox >= 24',
                                            'Explorer >= 8',
                                            'ios >= 5',
                                            'android >= 2.3',
                                            'bb >= 10',
                                            'last 1 version'
                                        ]
                                    })
                                ]
                            }
                        },
                        {
                            loader: 'sass-loader'
                        }

                    ]
                },
                {
                    test: /\.(jpg|png|gif|jpeg|woff|woff2|svg|ttf|eot)$/,
                    use: [

                        {
                            loader: 'url-loader',
                            options: {
                                limit: 8192,
                                name: 'images/[name]-[hash:32].[ext]'
                            }
                        }

                    ]
                },
                {
                    test: /\.html$/,
                    loader: 'html-withimg-loader'
                },
                {
                    test: /\.(woff|woff2|svg|ttf|eot)$/,
                    use: [

                        {
                            loader: 'file-loader',
                            options: {
                                name: 'fonts/[name].[hash:8].[ext]'
                            }
                        } // 项目设置打包到dist下的fonts文件夹下

                    ]
                }
            ]
        },
        plugins: [
            new HtmlWebpackPlugin({
                minify: {
                    removeAttributeQuotes: true // 去掉属性值后的双引号
                },
                hash: true,
                template: './src/index.html',
                inject: 'body'
            }),
            new CleanWebpackPlugin(['dist']),
            new MiniCssExtractPlugin({
                filename: 'css/[name]-[hash:32].css'
            }),
            new OptimizeCssAssetsPlugin({
                assetNameRegExp: /\.css$/g,
                cssProcessor: require('cssnano'),
                cssProcessorOptions: {
                    safe: true,
                    discardComments: {
                        removeAll: true
                    }
                },
                canPrint: true
            }),
            new OpenBrowserPlugin({
                url: 'http://localhost:3000'
            }),
            new CompressionWebpackPlugin({
                test: new RegExp('\\.(' + productionGzipExtensions.join('|') + ')),
                    threshold: 8192,
                    minRatio: 0.8
                })],
            devServer: {
                contentBase: path.join(__dirname, 'dist'),
                compress: true,
                port: 3000
            }
        }
```
