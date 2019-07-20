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
	"github.com/emer/vision/gabor"
	"github.com/emer/vision/kwta"
	"github.com/emer/vision/vfilter"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
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
	V1sGabor    gabor.Filter     `desc:"V1 simple gabor filter parameters"`
	V1sGeom     vfilter.Geom     `inactive:"+" view:"inline" desc:"geometry of input, output for V1 simple-cell processing"`
	V1sKWTA     kwta.KWTA        `desc:"kwta parameters for V1s"`
	ImgSize     image.Point      `desc:"target image size to use -- images will be rescaled to this size"`
	V1sGaborTsr etensor.Float32  `view:"no-inline" desc:"V1 simple gabor filter tensor"`
	V1sGaborTab etable.Table     `view:"no-inline" desc:"V1 simple gabor filter table (view only)"`
	Img         image.Image      `view:"-" desc:"current input image"`
	ImgTsr      etensor.Float32  `view:"no-inline" desc:"input image as tensor"`
	V1sTsr      etensor.Float32  `view:"no-inline" desc:"V1 simple gabor filter output tensor"`
	V1sKwtaTsr  etensor.Float32  `view:"no-inline" desc:"V1 simple gabor filter output, kwta output tensor"`
	V1sPoolTsr  etensor.Float32  `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 tensor"`
	V1sInhibs   []kwta.FFFBInhib `view:"no-inline" desc:"inhibition values for V1s KWTA"`
}

func (vi *Vis) Defaults() {
	vi.V1sGabor.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.V1sGabor.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.V1sGeom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.V1sKWTA.Defaults()
	vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.V1sGabor.ToTensor(&vi.V1sGaborTsr)
	vi.V1sGabor.ToTable(&vi.V1sGaborTab) // note: view only, testing
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
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.V1sGeom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.V1sGeom.FiltRt.X)
	return nil
}

// V1Simple runs V1Simple Gabor filtering on input image
// must have valid Img in place to start.
// Runs kwta and pool steps after gabor filter.
func (vi *Vis) V1Simple() {
	vfilter.Conv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgTsr, &vi.V1sTsr, vi.V1sGabor.Gain)
	vi.V1sKWTA.KWTAPool(&vi.V1sTsr, &vi.V1sKwtaTsr, &vi.V1sInhibs)
	vfilter.MaxPool(image.Point{2, 2}, &vi.V1sKwtaTsr, &vi.V1sPoolTsr)
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
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
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

	vp.UpdateEndNoSig(updt)

	win.MainMenuUpdated()
	return win
}

var TheVis Vis

func mainrun() {
	TheVis.Defaults()
	err := TheVis.OpenImage("side-tee-128.png")
	//	err := TheVis.OpenImage("img_0001_p00_005_tablelamp_007_tick_5_sac_1.jpg")
	//	err := TheVis.OpenImage("img_0001_p00_005_tablelamp_007_tick_5_sac_1_crop.jpg")
	if err != nil {
		return
	}
	TheVis.V1Simple()
	win := TheVis.ConfigGui()
	win.StartEventLoop()
}
