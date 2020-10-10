Generative models (including ProGAN) running with tensorflow.js

https://josephrocca.github.io/tfjs-gans/index.html

# Setup

Download this repo and serve the contents of the folder with a static webserver of your choice. For example:

```sh
git clone https://github.com/josephrocca/tfjs-gans
cd tfjs-gans
# install deno if you don't have it: https://deno.land/
deno run --allow-net --allow-read=. https://raw.githubusercontent.com/josephrocca/denoSimpleStatic/master/main.ts
```

The server will say something like "Start listening on 127.0.0.1:8000", and so you'd open up `127.0.0.1:8000` in your browser.
