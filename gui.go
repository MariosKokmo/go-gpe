package main

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"strconv"
	"sync"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog" // Needed for mouse events
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
)

// --- AppUI Struct Definition  ---
type AppUI struct {
	App    fyne.App
	Window fyne.Window // The main window (passed in)

	// Widgets that need updating or interaction handling
	densityPlot *canvas.Raster
	phasePlot   *canvas.Raster
	timeLabel   *widget.Label
	gSlider     *widget.Slider
	gLabel      *widget.Label
	coordLabel  *widget.Label
	startButton *widget.Button
	stopButton  *widget.Button
	resetButton *widget.Button

	// Plotting state
	lastPlotData PlotData
	plotMutex    sync.Mutex
	plotWidth    int
	plotHeight   int

	// Communication channels with sim
	updateChan  <-chan PlotData
	controlChan chan<- ControlMsg

	// The main container holding the final UI content
	Container fyne.CanvasObject
}

// setupMainUI creates the main simulation UI content (plots, controls)
// and configures the passed-in mainWindow. It starts the GUI update loop.
func setupMainUI(a fyne.App, initialParams SimParams, updateChan <-chan PlotData, controlChan chan<- ControlMsg, mainWindow fyne.Window) *AppUI {
	log.Println("Setting up main UI content...")

	// Initialize the AppUI struct, crucially storing the passed-in mainWindow.
	ui := &AppUI{
		App:         a,
		Window:      mainWindow, // Use the window passed from main
		updateChan:  updateChan,
		controlChan: controlChan,
		plotWidth:   256, // Initial default plot size guess
		plotHeight:  256,
		// lastPlotData is initially empty
	}

	// --- Create Widgets (same logic as previous createMainWindow) ---

	// Time Label
	ui.timeLabel = widget.NewLabel("Time: 0.000")
	ui.timeLabel.Alignment = fyne.TextAlignTrailing

	// G Slider and Label
	defaultG := 1.0 // Or get from initialParams if needed, though usually starts fixed
	ui.gLabel = widget.NewLabel(fmt.Sprintf("%.1f", defaultG))
	ui.gLabel.Alignment = fyne.TextAlignLeading
	ui.gSlider = widget.NewSlider(-100.0, 1000.0)
	ui.gSlider.SetValue(defaultG)
	ui.gSlider.Step = 10.0
	ui.gSlider.OnChanged = func(val float64) {
		ui.gLabel.SetText(fmt.Sprintf("%.1f", val))
		select {
		case ui.controlChan <- ControlMsg{Command: "update_g", GValue: val}:
		default: log.Println("Control channel full, cannot send g update.")
		}
	}

	// Coordinate Label
	ui.coordLabel = widget.NewLabel("Coords: (---, ---)")

	// Control Buttons
	ui.startButton = widget.NewButton("Start", func() {
		log.Println("Start button pressed.")
		select {
		case ui.controlChan <- ControlMsg{Command: "start"}:
			log.Println("Start command sent to simulation.")
		default:
			log.Println("Control channel full, cannot send start command.")
		}
	})
	ui.stopButton = widget.NewButton("Stop", func() {
		select {
		case ui.controlChan <- ControlMsg{Command: "stop"}:
		default: log.Println("Control channel full, cannot send stop.")
		}
	})
	ui.resetButton = widget.NewButton("Reset", func() {
		currentG := ui.gSlider.Value
		select {
		case ui.controlChan <- ControlMsg{Command: "reset", GValue: currentG}:
		default: log.Println("Control channel full, cannot send reset.")
		}
		ui.gSlider.SetValue(currentG) // Ensure slider reflects the value sent
		ui.gLabel.SetText(fmt.Sprintf("%.1f", currentG))
		ui.timeLabel.SetText("Time: 0.000")
	})

	// --- Plotting Area ---
	ui.densityPlot = canvas.NewRaster(ui.drawDensity) // Generator defined below (or in AppUI methods)
	ui.densityPlot.SetMinSize(fyne.NewSize(float32(ui.plotWidth), float32(ui.plotHeight)))

	ui.phasePlot = canvas.NewRaster(ui.drawPhase) // Generator defined below
	ui.phasePlot.SetMinSize(fyne.NewSize(float32(ui.plotWidth), float32(ui.plotHeight)))

	// --- Interaction Handling ---
	densityInteractionLayer := NewTappableRaster(ui.densityPlot, ui.handlePlotTap, ui.handlePlotHover)

	// --- Layout (same logic as previous createMainWindow) ---
	gControl := container.NewBorder(nil, nil, widget.NewLabel("Interaction g:"), ui.gLabel, ui.gSlider)
	buttons := container.NewHBox(ui.startButton, ui.stopButton, ui.resetButton)
	statusLabels := container.NewBorder(nil, nil, ui.coordLabel, ui.timeLabel)
	controls := container.NewVBox(buttons, gControl, statusLabels)

	densityPlotWithLabel := container.NewVBox(
		ui.densityPlot,
		densityInteractionLayer,
		container.New(layout.NewPaddedLayout(),
			container.NewVBox(widget.NewLabelWithStyle("Density (z=0) - Click to Place Vortex", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}), layout.NewSpacer())),
	)
	phasePlotWithLabel := container.NewVBox(
		ui.phasePlot,
		container.New(layout.NewPaddedLayout(),
			container.NewVBox(widget.NewLabelWithStyle("Phase (z=0)", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}), layout.NewSpacer())),
	)
	plots := container.NewGridWithColumns(2, densityPlotWithLabel, phasePlotWithLabel)

	// --- Create the final content container ---
	// This is the main layout that will be set on the window in `main`.
	finalContent := container.NewBorder(nil, controls, nil, nil, plots)
	ui.Container = finalContent // Store the final content in the AppUI struct

	// --- Start GUI update loop ---
	// Launch the goroutine responsible for receiving data and updating the UI.
	// This needs access to the `ui` struct to update widgets and manage state.
	go ui.guiUpdateLoop()
	log.Println("GUI update loop started from setupMainUI.")

	log.Println("Main UI content setup finished.")
	return ui // Return the AppUI struct containing the window, content, and widgets
}

// --- Methods attached to AppUI (needed by setupMainUI and its callbacks) ---

// guiUpdateLoop runs in a separate goroutine, listening for simulation updates.
func (ui *AppUI) guiUpdateLoop() {
    log.Println("GUI update loop started.")
    for plotData := range ui.updateChan {
        //log.Println("Received plot data update.")
        if plotData.Error != nil {
            log.Printf("Error in plot data: %v", plotData.Error)
            if ui.Window != nil {
                dialog.ShowError(plotData.Error, ui.Window)
            }
            continue
        }

        ui.plotMutex.Lock()
        ui.lastPlotData = plotData
        ui.plotMutex.Unlock()

        if ui.timeLabel != nil {
            ui.timeLabel.SetText(fmt.Sprintf("Time: %.3f", plotData.Time))
        }
        if ui.densityPlot != nil {
            ui.densityPlot.Refresh()
        }
        if ui.phasePlot != nil {
            ui.phasePlot.Refresh()
        }
        //log.Println("Plot data updated.")
    }
    log.Println("GUI update loop finished (update channel closed).")
}

// drawDensity generates the image for the density plot. (Attached to AppUI)
func (ui *AppUI) drawDensity(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	ui.plotWidth, ui.plotHeight = w, h

	ui.plotMutex.Lock()
	data := ui.lastPlotData.DensitySlice
	ui.plotMutex.Unlock()

	if data == nil || len(data) == 0 {
		// Check if inner slice exists and has length before accessing len(data[0])
        drawPlaceholder(img, color.NRGBA{R: 20, G: 20, B: 40, A: 255})
        return img
    }
    if len(data[0]) == 0{
         drawPlaceholder(img, color.NRGBA{R: 20, G: 20, B: 40, A: 255})
        return img
    }

	rows := len(data)
	cols := len(data[0])

	maxDensity := 0.0
	hasFiniteData := false
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			d := data[r][c]
			if !math.IsNaN(d) && !math.IsInf(d, 0) {
				hasFiniteData = true
				if d > maxDensity { maxDensity = d }
			}
		}
	}
	colorScaleMax := maxDensity * 1.1
	if !hasFiniteData { colorScaleMax = 0.1 } else if colorScaleMax < 1e-9 { colorScaleMax = 0.1 }


	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dataR := int(float64(y) * float64(rows) / float64(h))
			dataC := int(float64(x) * float64(cols) / float64(w))

			if dataR >= 0 && dataR < rows && dataC >= 0 && dataC < cols {
				val := data[dataR][dataC]
				if math.IsNaN(val) || math.IsInf(val, 0) { img.Set(x, y, color.RGBA{R: 255, G: 0, B: 255, A: 255}) } else {
					normVal := math.Max(0, math.Min(1, val/colorScaleMax))
					img.Set(x, y, viridis(normVal)) // Make sure viridis is defined
				}
			} else { img.Set(x, y, color.Black) }
		}
	}
	return img
}

// drawPhase generates the image for the phase plot. (Attached to AppUI)
func (ui *AppUI) drawPhase(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))

	ui.plotMutex.Lock()
	data := ui.lastPlotData.PhaseSlice
	ui.plotMutex.Unlock()

	if data == nil || len(data) == 0 {
        drawPlaceholder(img, color.NRGBA{R: 40, G: 20, B: 20, A: 255})
        return img
    }
    if len(data[0]) == 0 {
         drawPlaceholder(img, color.NRGBA{R: 40, G: 20, B: 20, A: 255})
        return img
    }


	rows := len(data)
	cols := len(data[0])

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dataR := int(float64(y) * float64(rows) / float64(h))
			dataC := int(float64(x) * float64(cols) / float64(w))

			if dataR >= 0 && dataR < rows && dataC >= 0 && dataC < cols {
				val := data[dataR][dataC]
				if math.IsNaN(val) || math.IsInf(val, 0) { img.Set(x, y, color.RGBA{R: 255, G: 0, B: 255, A: 255}) } else {
					normPhase := (val + math.Pi) / (2 * math.Pi)
					img.Set(x, y, hsv(normPhase)) // Make sure hsv is defined
				}
			} else { img.Set(x, y, color.Black) }
		}
	}
	return img
}

// handlePlotTap callback for vortex imprint. (Attached to AppUI)
func (ui *AppUI) handlePlotTap(pos fyne.Position) {
	ui.plotMutex.Lock()
	extent := ui.lastPlotData.Extent
	validData := ui.lastPlotData.DensitySlice != nil
	ui.plotMutex.Unlock()

	if !validData || (extent == [4]float64{}) { return }

	plotWidth := float32(ui.plotWidth)
	plotHeight := float32(ui.plotHeight)
	if plotWidth < 1 || plotHeight < 1 { return }

	simX := extent[0] + (float64(pos.X/plotWidth) * (extent[1] - extent[0]))
	simY := extent[2] + (float64(pos.Y/plotHeight)*(extent[3] - extent[2]))
	//simY := extent[2] + (float64((plotHeight-pos.Y)/plotHeight) * (extent[3] - extent[2]))

	chargeEntry := widget.NewEntry(); chargeEntry.SetText("1")
	chargeEntry.Validator = func(s string) error { _, err := strconv.Atoi(s); if err != nil { return errors.New("charge must be an integer") }; return nil }
	items := []*widget.FormItem{widget.NewFormItem("Charge", chargeEntry)}

	dialog.ShowForm(fmt.Sprintf("Imprint Vortex at (%.2f, %.2f)", simX, simY), "OK", "Cancel", items,
		func(ok bool) {
			if !ok { return }
			chargeStr := chargeEntry.Text
			charge, err := strconv.Atoi(chargeStr)
			if err != nil { dialog.ShowError(fmt.Errorf("invalid charge: %v", err), ui.Window); return }

			vortexInfo := VortexInfo{X0: simX, Y0: simY, Charge: charge}
			select {
			case ui.controlChan <- ControlMsg{Command: "imprint", Vortex: vortexInfo}:
				log.Printf("Sent imprint request: q=%d at (%.2f, %.2f)", charge, simX, simY)
			default:
				log.Println("Control channel full, cannot send imprint command.")
				dialog.ShowInformation("Busy", "Simulation is busy, cannot imprint vortex now.", ui.Window)
			}
		}, ui.Window)
}

// handlePlotHover callback for coordinate display. (Attached to AppUI)
func (ui *AppUI) handlePlotHover(pos fyne.Position, exited bool) {
	if exited {
		if ui.coordLabel != nil { ui.coordLabel.SetText("Coords: (---, ---)") }
		return
	}
	if ui.coordLabel == nil { return } // Guard against nil widget

	ui.plotMutex.Lock()
	extent := ui.lastPlotData.Extent
	validData := ui.lastPlotData.DensitySlice != nil
	ui.plotMutex.Unlock()

	if !validData || (extent == [4]float64{}) { return }

	plotWidth := float32(ui.plotWidth); plotHeight := float32(ui.plotHeight)
	if plotWidth < 1 || plotHeight < 1 { return }

	simX := extent[0] + (float64(pos.X/plotWidth) * (extent[1] - extent[0]))
	simY := extent[2] + (float64((plotHeight-pos.Y)/plotHeight) * (extent[3] - extent[2]))
	ui.coordLabel.SetText(fmt.Sprintf("Coords: (%.2f, %.2f)", simX, simY))
}
