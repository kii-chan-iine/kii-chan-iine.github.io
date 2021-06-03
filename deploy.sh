#!/usr/bin/env sh

# 确保脚本抛出遇到的错误
set -e

# 生成静态文件
# npm run build


date=`date +"%Y-%m-%d %H:%M:%S"`
commitmsg='Site updated: '$date
# yarn build
git add -A
git commit -m "$commitmsg"
git config --local user.name kii-chan-iine
git config --local user.email kaichen1993@hotmail.com

# 如果发布到 https://<USERNAME>.github.io
git push -f git@github.com:kii-chan-iine/kii-chan-iine.github.io.git
