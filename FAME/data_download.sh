dir="data"
file="data.zip"
if [ -d "$dir" ]; then
	echo "$dir found."
else
	url="https://drive.google.com/uc?id=1z0-MpDrmYh5Gi17x4ZRbavsEaXIyu4oZ&export=download"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
    unzip "${file}"
    rm "${file}"
fi
