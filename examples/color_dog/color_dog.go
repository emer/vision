// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"log"

	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/core"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor/tensorcore"
	"cogentcore.org/core/tree"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/vision/v2/colorspace"
	"github.com/emer/vision/v2/dog"
	"github.com/emer/vision/v2/vfilter"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add

	// name of image file to operate on -- if macbeth or empty use the macbeth standard color test image
	ImageFile core.Filename

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
	DoGTsr tensor.Float32 `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// current input image
	Img image.Image `display:"-"`

	// input image as RGB tensor
	ImgTsr tensor.Float32 `display:"no-inline"`

	// LMS components + opponents tensor version of image
	ImgLMS tensor.Float32 `display:"no-inline"`

	// output from 3 dogs with different tuning
	OutAll tensor.Float32 `display:"no-inline"`

	// DoG filter output tensors
	OutTsrs map[string]*tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ImageFile = ""                          // core.Filename("GrangerRainbow.png")
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
	vi.DoGTab.Init()
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	tensorcore.AddGridStylerTo(vi.DoGTab.Columns.Values[1], func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.01, 0.01)
	})
	vi.OutTsrs = make(map[string]*tensor.Float32)
	tensorcore.AddGridStylerTo(&vi.ImgTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.ImgLMS, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
}

// OutTsr gets output tensor of given name, creating if not yet made
func (vi *Vis) OutTsr(name string) *tensor.Float32 {
	if vi.OutTsrs == nil {
		vi.OutTsrs = make(map[string]*tensor.Float32)
	}
	tsr, ok := vi.OutTsrs[name]
	if !ok {
		tsr = &tensor.Float32{}
		vi.OutTsrs[name] = tsr
		tensorcore.AddGridStylerTo(tsr, func(s *tensorcore.GridStyle) {
			s.GridFill = 1
		})
	}
	return tsr
}

// OpenImage opens given filename as current image Img
func (vi *Vis) OpenImage(filepath string) error { //types:add
	var err error
	vi.Img, _, err = imagex.Open(filepath)
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
	return nil
}

// OpenMacbeth opens the macbeth test image
func (vi *Vis) OpenMacbeth() error {
	colorspace.MacbethImage(&vi.ImgTsr, vi.ImgSize.X, vi.ImgSize.Y, vi.Geom.FiltRt.X)
	colorspace.RGBTensorToLMSComps(&vi.ImgLMS, &vi.ImgTsr)
	img := &image.RGBA{}
	img = vfilter.RGBTensorToImage(img, &vi.ImgTsr, 0, false)
	vi.Img = img
	var err error
	err = imagex.Save(img, "macbeth.png")
	if err != nil {
		log.Println(err)
		return err
	}
	return nil
}

// ColorDoG runs color contrast DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) ColorDoG() {
	rimg := vi.ImgLMS.SubSpace(int(colorspace.LC)).(*tensor.Float32)
	gimg := vi.ImgLMS.SubSpace(int(colorspace.MC)).(*tensor.Float32)
	tensorcore.AddGridStylerTo(rimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	tensorcore.AddGridStylerTo(gimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	vi.OutTsrs["Red"] = rimg
	vi.OutTsrs["Green"] = gimg

	bimg := vi.ImgLMS.SubSpace(int(colorspace.SC)).(*tensor.Float32)
	yimg := vi.ImgLMS.SubSpace(int(colorspace.LMC)).(*tensor.Float32)
	tensorcore.AddGridStylerTo(bimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	tensorcore.AddGridStylerTo(yimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	vi.OutTsrs["Blue"] = bimg
	vi.OutTsrs["Yellow"] = yimg

	// for display purposes only:
	byimg := vi.ImgLMS.SubSpace(int(colorspace.SvLMC)).(*tensor.Float32)
	rgimg := vi.ImgLMS.SubSpace(int(colorspace.LvMC)).(*tensor.Float32)
	tensorcore.AddGridStylerTo(byimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	tensorcore.AddGridStylerTo(rgimg, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
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
	ny := otsr.DimSize(1)
	nx := otsr.DimSize(2)
	vi.OutAll.SetShapeSizes(ny, nx, 2, 2*len(vi.DoGNames))
	tensorcore.AddGridStylerTo(&vi.OutAll, func(s *tensorcore.GridStyle) {
		s.GridFill = 1
	})
	for i, nm := range vi.DoGNames {
		rgtsr := vi.OutTsr("DoG_" + nm + "_Red-Green")
		bytsr := vi.OutTsr("DoG_" + nm + "_Blue-Yellow")
		vfilter.OuterAgg(i*2, 0, rgtsr, &vi.OutAll)
		vfilter.OuterAgg(i*2+1, 0, bytsr, &vi.OutAll)
	}
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
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

func (vi *Vis) ConfigGUI() *core.Body {
	b := core.NewBody("colordog").SetTitle("Color DoGFiltering")
	core.NewForm(b).SetStruct(vi)
	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
