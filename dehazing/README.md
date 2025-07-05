# Dehazing

## data

```sh
aria2c -x10 -s10 -k1M http://www.vision.ee.ethz.ch/ntire18/i-haze/I-HAZE.zip
unzip I-HAZE.zip
mv "# I-HAZY NTIRE 2018" data
rm I-HAZE.zip
```

# CVPR2009 usage

```sh
uv run python cvpr2009.py --input data/hazy/01_indoor_hazy.jpg --output cvpr2009.jpg
```

# CinvIR

```sh
gdown 1Hnu-RKoP3IqZDANeP_p0AKKHn1B4Yl3Y
```

run

```sh
uv run python ConvIR/Dehazing/OTS/main.py --mode infer --type base --test_model ots-base.pkl --input_image data/hazy/01_indoor_hazy.jpg --output_image ConvIR.jpg
```
