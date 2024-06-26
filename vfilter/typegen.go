// Code generated by "core generate -add-types"; DO NOT EDIT.

package vfilter

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "github.com/emer/vision/v2/vfilter.Geom", IDName: "geom", Doc: "Geom contains the filtering geometry info for a given filter pass.", Fields: []types.Field{{Name: "In", Doc: "size of input -- computed from image or set"}, {Name: "Out", Doc: "size of output -- computed"}, {Name: "Border", Doc: "starting border into image -- must be >= FiltRt"}, {Name: "Spacing", Doc: "spacing -- number of pixels to skip in each direction"}, {Name: "FiltSz", Doc: "full size of filter"}, {Name: "FiltLt", Doc: "computed size of left/top size of filter"}, {Name: "FiltRt", Doc: "computed size of right/bottom size of filter (FiltSz - FiltLeft)"}}})
