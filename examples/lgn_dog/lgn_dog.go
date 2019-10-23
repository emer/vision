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
	ImageFile gi.FileName     `desc:"name of image file to operate on"`
	DoG       dog.Filter      `desc:"LGN DoG filter parameters"`
	Geom      vfilter.Geom    `inactive:"+" view:"inline" desc:"geometry of input, output"`
	ImgSize   image.Point     `desc:"target image size to use -- images will be rescaled to this size"`
	DoGTsr    etensor.Float32 `view:"no-inline" desc:"DoG filter tensor -- has 3 filters (on, off, net)"`
	DoGTab    etable.Table    `view:"no-inline" desc:"DoG filter table (view only)"`
	Img       image.Image     `view:"-" desc:"current input image"`
	ImgTsr    etensor.Float32 `view:"no-inline" desc:"input image as tensor"`
	OutTsr    etensor.Float32 `view:"no-inline" desc:"DoG filter output tensor"`
}

var KiT_Vis = kit.Types.AddType(&Vis{}, VisProps)

func (vi *Vis) Defaults() {
	vi.ImageFile = gi.FileName("side-tee-128.png")
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.DoG.ToTensor(&vi.DoGTsr)
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	vi.DoGTab.Cols[1].SetMetaData("max", "0.2")
	vi.DoGTab.Cols[1].SetMetaData("min", "-0.2")
}

// OpenImage opens given filename as current image Img
// and converts to a float32 tensor for processing
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
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.Geom.FiltRt.X)
	vi.ImgTsr.SetMetaData("image", "+")
	return nil
}

// LGNDoG runs DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Net)
	vfilter.Conv1(&vi.Geom, flt, &vi.ImgTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	vfilter.TensorLogNorm32(&vi.OutTsr, 0) // 0 = renorm all, 1 = renorm within each on / off separately
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error {
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		log.Println(err)
		return err
	}
	vi.LGNDoG()
	return nil
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this Vis
func (vi *Vis) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("v1gabor")
	gi.SetAppAbout(`This demonstrates basic V1 Gabor Filtering.  See <a href="https://github.com/emer/vision/v1">V1 on GitHub</a>.</p>`)

	win := gi.NewWindow2D("v1gabor", "V1 Gabor Filtering", width, height, true)
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
