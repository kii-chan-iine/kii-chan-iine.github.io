#!/usr/bin/env sh

# 确保脚本抛出遇到的错误
set -e

date=`date +"%Y-%m-%d %H:%M:%S"`
commitmsg='Site updated: '$date
# yarn build
git add -A
git commit -m "$commitmsg"
git config --local user.name vxhly
git config --local user.email pengchengou@gmail.com

# 如果发布到 https://<USERNAME>.github.io
git push -f git@github.com:vxhly/vxhly.github.io.git
