# @echo off

npm run build
# pause


# @echo off

cd ../develop\

cp -ru ./published-pages/* ./kii-chan-iine.github.io\

cd ./kii-chan-iine.github.io\

./pushall.bat
# pause