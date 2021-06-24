// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"
	"log"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/vision/colorspace"
	"github.com/emer/vision/dog"
	"github.com/emer/vision/vfilter"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// this is the stub main for gogi that calls our actual
// mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct {
	ImageFile  gi.FileName                 `desc:"name of image file to operate on -- if macbeth or empty use the macbeth standard color test image"`
	DoG        dog.Filter                  `desc:"LGN DoG filter parameters"`
	DoGNames   []string                    `desc:"names of the dog gain sets -- for naming output data"`
	DoGGains   []float32                   `desc:"overall gain factors, to compensate for diffs in OnGains"`
	DoGOnGains []float32                   `desc:"OnGain factors -- 1 = perfect balance, otherwise has relative imbalance for capturing main effects"`
	Geom       vfilter.Geom                `inactive:"+" view:"inline" desc:"geometry of input, output"`
	ImgSize    image.Point                 `desc:"target image size to use -- images will be rescaled to this size"`
	DoGTsr     etensor.Float32             `view:"no-inline" desc:"DoG filter tensor -- has 3 filters (on, off, net)"`
	DoGTab     etable.Table                `view:"no-inline" desc:"DoG filter table (view only)"`
	Img        image.Image                 `view:"-" desc:"current input image"`
	ImgTsr     etensor.Float32             `view:"no-inline" desc:"input image as RGB tensor"`
	ImgLMS     etensor.Float32             `view:"no-inline" desc:"LMS components + opponents tensor version of image"`
	OutAll     etensor.Float32             `view:"no-inline" desc:"output from 3 dogs with different tuning"`
	OutTsrs    map[string]*etensor.Float32 `view:"no-inline" desc:"DoG filter output tensors"`
}

var KiT_Vis = kit.Types.AddType(&Vis{}, VisProps)

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
func (vi *Vis) OpenImage(filepath string) error {
	var err error
	vi.Img, err = gi.OpenImage(filepath)
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
	return nil
}

// OpenMacbeth opens the macbeth test image
func (vi *Vis) OpenMacbeth() error {
	colorspace.MacbethImage(&vi.ImgTsr, vi.ImgSize.X, vi.ImgSize.Y, vi.Geom.FiltRt.X)
	colorspace.RGBTensorToLMSComps(&vi.ImgLMS, &vi.ImgTsr)
	vi.ImgTsr.SetMetaData("image", "+")
	img := &image.RGBA{}
	img = vfilter.RGBTensorToImage(img, &vi.ImgTsr, 0, false)
	vi.Img = img
	var err error
	err = gi.SaveImage("macbeth.png", img)
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
func (vi *Vis) Filter() error {
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

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this Vis
func (vi *Vis) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("colordog")
	gi.SetAppAbout(`This demonstrates LMS colorspace difference-of-gaussian blob filtering.  See <a href="https://github.com/emer/vision">Vision on GitHub</a>.</p>`)

	win := gi.NewMainWindow("colordog", "Color DoGFiltering", width, height)
	// vi.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	// vi.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.Viewport = vp
	sv.SetStruct(vi)

	split.SetSplits(1)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	gi.SetQuitReqFunc(func() {
		gi.Quit()
	})
	win.SetCloseReqFunc(func(w *gi.Window) {
		gi.Quit()
	})
	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	vp.UpdateEndNoSig(updt)

	win.MainMenuUpdated()
	return win
}

// These props create interactive toolbar for GUI
var VisProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"Filter", ki.Props{
			"desc": "run filter methods on current ImageFile image",
			"icon": "updt",
		}},
	},
}

var TheVis Vis

func mainrun() {
	TheVis.Defaults()
	TheVis.Filter()
	win := TheVis.ConfigGui()
	win.StartEventLoop()
}
