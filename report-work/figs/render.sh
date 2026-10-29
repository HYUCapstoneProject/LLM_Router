#!/bin/zsh
# render.sh <name...>: name.svg or name.html -> name.png at 2x
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for n in "$@"; do
  f=$n.svg; [ -f $f ] || f=$n.html
  w=$(grep -o 'width="[0-9]*"' $f | head -1 | grep -o '[0-9]*'); h=$(grep -o 'height="[0-9]*"' $f | head -1 | grep -o '[0-9]*')
  if [ $f = $n.html ]; then w=$(grep -o 'data-w="[0-9]*"' $f | grep -o '[0-9]*'); h=$(grep -o 'data-h="[0-9]*"' $f | grep -o '[0-9]*'); fi
  "$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=$w,$h --screenshot=$PWD/$n.png file://$PWD/$f >/dev/null 2>&1
  echo "$n ${w}x${h}"
done
