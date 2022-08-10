@echo off
@set input1=
@set /p input1=ud_info:

git status
git add -A
git commit -a -m "%input1%"
git push

pause