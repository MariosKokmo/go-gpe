package main

import (
	"image"
	"image/color"
	"log"
	"math"
	"math/cmplx"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"
)

// TappableRaster definition and methods
type TappableRaster struct { widget.BaseWidget; raster *canvas.Raster; onTapped func(pos fyne.Position); onHovered func(pos fyne.Position, exited bool) }
func NewTappableRaster(raster *canvas.Raster, tapped func(fyne.Position), hovered func(fyne.Position, bool)) *TappableRaster { t := &TappableRaster{raster: raster, onTapped: tapped, onHovered: hovered}; t.ExtendBaseWidget(t); return t }
func (t *TappableRaster) CreateRenderer() fyne.WidgetRenderer { return widget.NewSimpleRenderer(t.raster) }
func (t *TappableRaster) Tapped(ev *fyne.PointEvent) { if t.onTapped != nil { t.onTapped(ev.Position) } }
func (t *TappableRaster) MouseOut() { if t.onHovered != nil { t.onHovered(fyne.Position{}, true) } }

// --- Data Extraction for Plotting ---
// These functions now take the wavefunction data as input, allowing them
// to operate on copies made outside the main simulation lock.

// getSliceParameters determines the slicing dimensions and indices.
// Helper function used by density/phase slicing.
func (sim *GPE3DSimulation) getSliceParameters(sliceAxis rune, sliceIndex int) (axis rune, index, rows, cols int) {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	axis = sliceAxis
	index = sliceIndex

	// Determine slice dimensions and default index based on the chosen axis
	switch axis {
	case 'x':
		if index < 0 || index >= nx { index = nx / 2 } // Default to middle index
		rows, cols = ny, nz // Slice is Ny (rows) x Nz (cols)
	case 'y':
		if index < 0 || index >= ny { index = ny / 2 }
		rows, cols = nx, nz // Slice is Nx (rows) x Nz (cols)
	case 'z':
		fallthrough // Default to 'z' axis
	default:
		axis = 'z'
		if index < 0 || index >= nz { index = nz / 2 }
		rows, cols = nx, ny // Slice is Nx (rows) x Ny (cols)
	}
	return axis, index, rows, cols
}

// getDensitySliceFromData extracts a 2D slice of the probability density |psi|^2
// from a provided wavefunction data array.
// It returns the 2D density slice and the plot extent [xmin, xmax, ymin, ymax].
func (sim *GPE3DSimulation) getDensitySliceFromData(psiData [][][]complex128, sliceAxis rune, sliceIndex int) ([][]float64, [4]float64) {
    if psiData == nil {
        log.Println("Error: getDensitySliceFromData called with nil psiData")
        return nil, [4]float64{}
    }
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	axis, index, rows, cols := sim.getSliceParameters(sliceAxis, sliceIndex)

	// Allocate the 2D slice for density results
	densitySlice := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		densitySlice[r] = make([]float64, cols)
	}

	// Extract data based on the slicing axis
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var val complex128
			switch axis {
			case 'x': // Slice plane YZ: rows=j=r, cols=k=c, fixed i=index
				if index >= 0 && index < nx && r >= 0 && r < ny && c >= 0 && c < nz {
					val = psiData[index][r][c]
				} else { val = complex(math.NaN(), math.NaN()) } // Mark invalid indices
			case 'y': // Slice plane XZ: rows=i=r, cols=k=c, fixed j=index
				if r >= 0 && r < nx && index >= 0 && index < ny && c >= 0 && c < nz {
					val = psiData[r][index][c]
				} else { val = complex(math.NaN(), math.NaN()) }
			case 'z': // Slice plane XY: rows=i=r, cols=j=c, fixed k=index
				if r >= 0 && r < nx && c >= 0 && c < ny && index >= 0 && index < nz {
					val = psiData[r][c][index] // Note: psiData[i][j][k] -> i=r, j=c
				} else { val = complex(math.NaN(), math.NaN()) }
			}
			// Calculate density |psi|^2. Use Abs for safety with NaNs.
            if cmplx.IsNaN(val) {
                 densitySlice[r][c] = math.NaN()
            } else {
			    absVal := cmplx.Abs(val)
			    densitySlice[r][c] = absVal * absVal
            }
		}
	}

	// Determine plot extent [xmin, xmax, ymin, ymax] based on the slice axis
	var extent [4]float64
	switch axis {
	case 'x': // Y vs Z plane. Plot shows Z horizontal (cols), Y vertical (rows).
		extent = [4]float64{sim.zEdges[0], sim.zEdges[1], sim.yEdges[0], sim.yEdges[1]}
        // Transpose needed if plotting expects Y horizontal, Z vertical
        densitySlice = transposeFloat64(densitySlice) // Transpose (Ny, Nz) -> (Nz, Ny)
        extent = [4]float64{sim.yEdges[0], sim.yEdges[1], sim.zEdges[0], sim.zEdges[1]} // Swap extent order after transpose
	case 'y': // X vs Z plane. Plot shows Z horizontal (cols), X vertical (rows).
		extent = [4]float64{sim.zEdges[0], sim.zEdges[1], sim.xEdges[0], sim.xEdges[1]}
        // Transpose needed if plotting expects X horizontal, Z vertical
        densitySlice = transposeFloat64(densitySlice) // Transpose (Nx, Nz) -> (Nz, Nx)
        extent = [4]float64{sim.xEdges[0], sim.xEdges[1], sim.zEdges[0], sim.zEdges[1]} // Swap extent order after transpose
	case 'z': // X vs Y plane. Plot shows Y horizontal (cols), X vertical (rows).
		extent = [4]float64{sim.yEdges[0], sim.yEdges[1], sim.xEdges[0], sim.xEdges[1]}
		// Transpose needed for conventional plot (X horizontal, Y vertical)
		densitySlice = transposeFloat64(densitySlice) // Transpose (Nx, Ny) -> (Ny, Nx)
		extent = [4]float64{sim.xEdges[0], sim.xEdges[1], sim.yEdges[0], sim.yEdges[1]} // Correct extent after transpose
	}

	//log.Println("Density slice extracted successfully.")
	return densitySlice, extent
}

// getPhaseSliceFromData extracts a 2D slice of the wavefunction phase angle(psi)
// from a provided wavefunction data array.
// It returns the 2D phase slice and the plot extent [xmin, xmax, ymin, ymax].
func (sim *GPE3DSimulation) getPhaseSliceFromData(psiData [][][]complex128, sliceAxis rune, sliceIndex int) ([][]float64, [4]float64) {
    if psiData == nil {
        log.Println("Error: getPhaseSliceFromData called with nil psiData")
        return nil, [4]float64{}
    }
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	axis, index, rows, cols := sim.getSliceParameters(sliceAxis, sliceIndex)

	// Allocate the 2D slice for phase results
	phaseSlice := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		phaseSlice[r] = make([]float64, cols)
	}

	// Extract data based on the slicing axis
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var val complex128
			switch axis {
			case 'x': // Slice plane YZ: rows=j=r, cols=k=c, fixed i=index
                 if index >= 0 && index < nx && r >= 0 && r < ny && c >= 0 && c < nz {
					val = psiData[index][r][c]
				} else { val = complex(math.NaN(), math.NaN()) }
			case 'y': // Slice plane XZ: rows=i=r, cols=k=c, fixed j=index
                 if r >= 0 && r < nx && index >= 0 && index < ny && c >= 0 && c < nz {
					val = psiData[r][index][c]
				} else { val = complex(math.NaN(), math.NaN()) }
			case 'z': // Slice plane XY: rows=i=r, cols=j=c, fixed k=index
                 if r >= 0 && r < nx && c >= 0 && c < ny && index >= 0 && index < nz {
					val = psiData[r][c][index]
				} else { val = complex(math.NaN(), math.NaN()) }
			}
			// Calculate phase angle. Returns NaN if input is NaN.
			phaseSlice[r][c] = cmplx.Phase(val)
		}
	}

	// Determine plot extent [xmin, xmax, ymin, ymax] based on the slice axis
	var extent [4]float64
	switch axis {
	case 'x': // Y vs Z plane -> Transpose -> Plot X=Y, Y=Z
		extent = [4]float64{sim.zEdges[0], sim.zEdges[1], sim.yEdges[0], sim.yEdges[1]}
		phaseSlice = transposeFloat64(phaseSlice)
		extent = [4]float64{sim.yEdges[0], sim.yEdges[1], sim.zEdges[0], sim.zEdges[1]}
	case 'y': // X vs Z plane -> Transpose -> Plot X=X, Y=Z
		extent = [4]float64{sim.zEdges[0], sim.zEdges[1], sim.xEdges[0], sim.xEdges[1]}
		phaseSlice = transposeFloat64(phaseSlice)
		extent = [4]float64{sim.xEdges[0], sim.xEdges[1], sim.zEdges[0], sim.zEdges[1]}
	case 'z': // X vs Y plane -> Transpose -> Plot X=X, Y=Y
		extent = [4]float64{sim.yEdges[0], sim.yEdges[1], sim.xEdges[0], sim.xEdges[1]}
		phaseSlice = transposeFloat64(phaseSlice)
		extent = [4]float64{sim.xEdges[0], sim.xEdges[1], sim.yEdges[0], sim.yEdges[1]}
	}

	//log.Println("Phase slice extracted successfully.")
	return phaseSlice, extent
}


// --- Utility functions (like transposeFloat64, flatten, unflatten) ---

// transposeFloat64 helper for transposing plot data slices (if needed by GUI).
func transposeFloat64(slice [][]float64) [][]float64 {
	if len(slice) == 0 { return slice }
	rows := len(slice)
	cols := len(slice[0])
	if cols == 0 { // Handle empty inner slices
		// Return slice with correct outer dimension but empty inner ones
		result := make([][]float64, cols) // cols is 0, so result is empty
        return result
	}

	// Allocate transposed slice
	result := make([][]float64, cols)
	for i := range result {
		result[i] = make([]float64, rows)
	}

	// Perform transpose
	for j := 0; j < rows; j++ { // Iterate original rows
		for i := 0; i < cols; i++ { // Iterate original columns
			result[i][j] = slice[j][i]
		}
	}
	return result
}


// Note: The `reset` functionality is primarily handled within the `handleControlMessage`
// function for the "reset" command case. It restores `sim.psi` from `sim.initialPsi`,
// resets `sim.time`, and clears `sim.vortices`.


// GUI Utility functions

// Placeholder for drawPlaceholder if not defined yet
func drawPlaceholder(img *image.RGBA, c color.Color) {
	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
}

// Placeholder for colormaps if not defined yet
func viridis(val float64) color.Color {
	// Simplified piecewise linear approximation of Viridis
	val = math.Max(0, math.Min(1, val)) // Clamp
	var r, g, b uint8
	if val < 0.25 { // Blue/Purple range
        f := val * 4.0
		r = uint8(68 + 1*f)
		g = uint8(1 + 55*f)
		b = uint8(84 + 51*f)
	} else if val < 0.5 { // Cyan/Green range
        f := (val - 0.25) * 4.0
		r = uint8(69 - 48*f)
		g = uint8(56 + 93*f)
		b = uint8(135 - 7*f)
	} else if val < 0.75 { // Yellow/Green range
        f := (val - 0.5) * 4.0
		r = uint8(21 + 108*f)
		g = uint8(149 + 51*f)
		b = uint8(128 - 93*f)
	} else { // Yellow range
        f := (val - 0.75) * 4.0
		r = uint8(129 + 126*f)
		g = uint8(200 + 23*f)
		b = uint8(35 - 31*f)
	}
	r = uint8(math.Max(0, math.Min(255, float64(r)))); g = uint8(math.Max(0, math.Min(255, float64(g)))); b = uint8(math.Max(0, math.Min(255, float64(b))))
	return color.NRGBA{R: r, G: g, B: b, A: 255}
}
func hsv(h float64) color.Color {
	h = math.Mod(h, 1.0); if h < 0 { h += 1.0 }
	i := math.Floor(h * 6); f := h*6 - i; const s, v = 1.0, 1.0; p := v * (1 - s); q := v * (1 - f*s); t := v * (1 - (1-f)*s)
	var r, g, b float64
	switch int(i) % 6 { case 0: r, g, b = v, t, p; case 1: r, g, b = q, v, p; case 2: r, g, b = p, v, t; case 3: r, g, b = p, q, v; case 4: r, g, b = t, p, v; case 5: r, g, b = v, p, q }; return color.NRGBA{R: uint8(r * 255), G: uint8(g * 255), B: uint8(b * 255), A: 255}
}
