// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate goki generate -add-types

import (
	"image"
	"log"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
	"github.com/emer/vision/v2/colorspace"
	"github.com/emer/vision/v2/dog"
	"github.com/emer/vision/v2/vfilter"
	"goki.dev/gi"
	"goki.dev/giv"
	"goki.dev/grows/images"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //gti:add

	// name of image file to operate on -- if macbeth or empty use the macbeth standard color test image
	ImageFile gi.FileName

	// LGN DoG filter parameters
	DoG dog.Filter

	// names of the dog gain sets -- for naming output data
	DoGNames []string

	// overall gain factors, to compensate for diffs in OnGains
	DoGGains []float32

	// OnGain factors -- 1 = perfect balance, otherwise has relative imbalance for capturing main effects
	DoGOnGains []float32

	// geometry of input, output
	Geom vfilter.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size
	ImgSize image.Point

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGTsr etensor.Float32 `view:"no-inline"`

	// DoG filter table (view only)
	DoGTab etable.Table `view:"no-inline"`

	// current input image
	Img image.Image `view:"-"`

	// input image as RGB tensor
	ImgTsr etensor.Float32 `view:"no-inline"`

	// LMS components + opponents tensor version of image
	ImgLMS etensor.Float32 `view:"no-inline"`

	// output from 3 dogs with different tuning
	OutAll etensor.Float32 `view:"no-inline"`

	// DoG filter output tensors
	OutTsrs map[string]*etensor.Float32 `view:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ImageFile = ""                          // gi.FileName("GrangerRainbow.png")
	vi.DoGNames = []string{"Bal", "On", "Off"} // balanced, gain toward On, gain toward Off
	vi.DoGGains = []float32{8, 4.1, 4.4}
	vi.DoGOnGains = []float32{1, 1.2, 0.833}
	sz := 16
	spc := 16
	vi.DoG.Defaults()
	vi.DoG.SetSize(sz, spc)
	vi.DoG.OnSig = .5 // no spatial component, just pure contrast
	vi.DoG.OffSig = .5
	vi.DoG.Gain = 8
	vi.DoG.OnGain = 1

	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImgSize = image.Point{512, 512}
	// vi.ImgSize = image.Point{256, 256}
	// vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.DoG.ToTensor(&vi.DoGTsr)
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	vi.DoGTab.Cols[1].SetMetaData("max", "0.2")
	vi.DoGTab.Cols[1].SetMetaData("min", "-0.2")
	vi.OutTsrs = make(map[string]*etensor.Float32)
}

// OutTsr gets output tensor of given name, creating if not yet made
func (vi *Vis) OutTsr(name string) *etensor.Float32 {
	if vi.OutTsrs == nil {
		vi.OutTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := vi.OutTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		vi.OutTsrs[name] = tsr
		tsr.SetMetaData("grid-fill", "1")
	}
	return tsr
}

// OpenImage opens given filename as current image Img
func (vi *Vis) OpenImage(filepath string) error { //gti:add
	var err error
	vi.Img, _, err = images.Open(filepath)
	if err != nil {
		log.Println(err)
		return err
	}
	isz := vi.Img.Bounds().Size()
	if isz != vi.ImgSize {
		vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
	}
	vfilter.RGBToTensor(vi.Img, &vi.ImgTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPadRGB(&vi.ImgTsr, vi.Geom.FiltRt.X)
	colorspace.RGBTensorToLMSComps(&vi.ImgLMS, &vi.ImgTsr)
	vi.ImgTsr.SetMetaData("image", "+")
	vi.ImgTsr.SetMetaData("min", "0")
	return nil
}

// OpenMacbeth opens the macbeth test image
func (vi *Vis) OpenMacbeth() error {
	colorspace.MacbethImage(&vi.ImgTsr, vi.ImgSize.X, vi.ImgSize.Y, vi.Geom.FiltRt.X)
	colorspace.RGBTensorToLMSComps(&vi.ImgLMS, &vi.ImgTsr)
	vi.ImgTsr.SetMetaData("image", "+")
	vi.ImgTsr.SetMetaData("min", "0")
	img := &image.RGBA{}
	img = vfilter.RGBTensorToImage(img, &vi.ImgTsr, 0, false)
	vi.Img = img
	var err error
	err = images.Save(img, "macbeth.png")
	if err != nil {
		log.Println(err)
		return err
	}
	return nil
}

// ColorDoG runs color contrast DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) ColorDoG() {
	rimg := vi.ImgLMS.SubSpace([]int{int(colorspace.LC)}).(*etensor.Float32)
	gimg := vi.ImgLMS.SubSpace([]int{int(colorspace.MC)}).(*etensor.Float32)
	rimg.SetMetaData("grid-fill", "1")
	gimg.SetMetaData("grid-fill", "1")
	vi.OutTsrs["Red"] = rimg
	vi.OutTsrs["Green"] = gimg

	bimg := vi.ImgLMS.SubSpace([]int{int(colorspace.SC)}).(*etensor.Float32)
	yimg := vi.ImgLMS.SubSpace([]int{int(colorspace.LMC)}).(*etensor.Float32)
	bimg.SetMetaData("grid-fill", "1")
	yimg.SetMetaData("grid-fill", "1")
	vi.OutTsrs["Blue"] = bimg
	vi.OutTsrs["Yellow"] = yimg

	// for display purposes only:
	byimg := vi.ImgLMS.SubSpace([]int{int(colorspace.SvLMC)}).(*etensor.Float32)
	rgimg := vi.ImgLMS.SubSpace([]int{int(colorspace.LvMC)}).(*etensor.Float32)
	byimg.SetMetaData("grid-fill", "1")
	rgimg.SetMetaData("grid-fill", "1")
	vi.OutTsrs["Blue-Yellow"] = byimg
	vi.OutTsrs["Red-Green"] = rgimg

	for i, nm := range vi.DoGNames {
		vi.DoGFilter(nm, vi.DoGGains[i], vi.DoGOnGains[i])
	}
}

// DoGFilter runs filtering for given gain factors
func (vi *Vis) DoGFilter(name string, gain, onGain float32) {
	dogOn := vi.DoG.FilterTensor(&vi.DoGTsr, dog.On)
	dogOff := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Off)

	rgtsr := vi.OutTsr("DoG_" + name + "_Red-Green")
	rimg := vi.OutTsr("Red")
	gimg := vi.OutTsr("Green")
	vfilter.ConvDiff(&vi.Geom, dogOn, dogOff, rimg, gimg, rgtsr, gain, onGain)

	bytsr := vi.OutTsr("DoG_" + name + "_Blue-Yellow")
	bimg := vi.OutTsr("Blue")
	yimg := vi.OutTsr("Yellow")
	vfilter.ConvDiff(&vi.Geom, dogOn, dogOff, bimg, yimg, bytsr, gain, onGain)
}

// AggAll aggregates the different DoG components into
func (vi *Vis) AggAll() {
	otsr := vi.OutTsr("DoG_" + vi.DoGNames[0] + "_Red-Green")
	ny := otsr.Dim(1)
	nx := otsr.Dim(2)
	oshp := []int{ny, nx, 2, 2 * len(vi.DoGNames)}
	vi.OutAll.SetShape(oshp, nil, []string{"Y", "X", "OnOff", "RGBY"})
	vi.OutAll.SetMetaData("grid-fill", "1")
	for i, nm := range vi.DoGNames {
		rgtsr := vi.OutTsr("DoG_" + nm + "_Red-Green")
		bytsr := vi.OutTsr("DoG_" + nm + "_Blue-Yellow")
		vfilter.OuterAgg(i*2, 0, rgtsr, &vi.OutAll)
		vfilter.OuterAgg(i*2+1, 0, bytsr, &vi.OutAll)
	}
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //gti:add
	if vi.ImageFile == "" || vi.ImageFile == "macbeth" {
		err := vi.OpenMacbeth()
		if err != nil {
			log.Println(err)
			return err
		}
	} else {
		err := vi.OpenImage(string(vi.ImageFile))
		if err != nil {
			log.Println(err)
			return err
		}
	}
	vi.ColorDoG()
	vi.AggAll()
	return nil
}

//////////////////////////////////////////////////////////////////////////////
// 		Gui

func (vi *Vis) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("colordog").SetTitle("Color DoGFiltering")
	b.App().About = `This demonstrates LMS colorspace difference-of-gaussian blob filtering.  See <a href="https://github.com/emer/vision">Vision on GitHub</a>.</p>`

	giv.NewStructView(b, "sv").SetStruct(vi)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, vi.Filter)
		// gi.NewSeparator(tb)
		// vi.Img.ConfigToolbar(tb)
		// gi.NewSeparator(tb)
		// vi.OutAll.ConfigToolbar(tb)
	})

	b.NewWindow().Run().Wait()
	return b
}
