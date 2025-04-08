package main

import (
	"errors"
	"fmt"
	"log"
	"strconv"
	"time"

	// Needed for waiting on simulation shutdown
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"
	// We will need other imports later for GUI/Simulation parts
)

// SimParams holds the configuration parameters for the simulation.
// Corresponds to the dictionary returned by get_simulation_parameters in Python.
type SimParams struct {
	Nx, Ny, Nz int     // Grid resolution
	L          float64 // Axis Length
	OmegaX     float64 // Potential frequency X
	OmegaY     float64 // Potential frequency Y
	OmegaZ     float64 // Potential frequency Z
}


// main is the entry point of the application.
func main() {
	log.Println("Starting application...")

	myApp := app.New()

	// 1. Create the main window.
	simWindow := myApp.NewWindow("GPE Simulation Setup")
	simWindow.SetContent(container.NewCenter(widget.NewLabel("Loading configuration...")))
	simWindow.Resize(fyne.NewSize(400, 200)) // Small initial size
	simWindow.CenterOnScreen()
	simWindow.Show()

	// 3. Define default parameters.
	defaultParams := SimParams{Nx: 64, Ny: 64, Nz: 64, L: 20.0, OmegaX: 1.0, OmegaY: 1.0, OmegaZ: 1.0}

	// 4. Show the configuration dialog, now using the simWindow as parent.
	showConfigDialog(myApp, simWindow, defaultParams, func(finalParams *SimParams, ok bool) {
		if !ok {
			log.Println("Simulation setup cancelled. Exiting.")
			simWindow.Close() // Close the initial window
			return
		}

		// --- Configuration OK ---
		log.Printf("Configuration successful: Grid=%dx%dx%d, L=%.2f, Omegas=(%.2f, %.2f, %.2f)",
			finalParams.Nx, finalParams.Ny, finalParams.Nz, finalParams.L,
			finalParams.OmegaX, finalParams.OmegaY, finalParams.OmegaZ)

		// 6. Setup Simulation
		updateChan := make(chan PlotData, 100)
		controlChan := make(chan ControlMsg, 50)
		sim, err := NewSimulation(*finalParams, updateChan, controlChan)
		if err != nil {
			log.Printf("Error initializing simulation: %v", err)
			dialog.ShowError(fmt.Errorf("Failed to initialize simulation:\n%v", err), simWindow)
			return
		}
		log.Println("Simulation engine initialized.")
		sim.wg.Add(1)
		go sim.Run()

		// 7. Setup the *final* UI content
		log.Println("Setting up main UI content...")
		ui := setupMainUI(myApp, sim.params, updateChan, controlChan, simWindow)
		if ui == nil || ui.Window == nil || ui.Container == nil {
			log.Println("Error: Final UI setup failed")
			dialog.ShowError(errors.New("Failed to create main application UI"), simWindow)
			sim.cancel()
			return
		}
		log.Println("Main UI created successfully.")

		// 8. Set the final content and configure the main window
		simWindow.SetTitle(fmt.Sprintf("Go 3D GPE Simulation (CPU) - %dx%dx%d",
			finalParams.Nx, finalParams.Ny, finalParams.Nz)) // Update title
		log.Println("Setting main window content...")
		simWindow.SetContent(ui.Container) // Set the real UI
		log.Println("Main window content set.")
		simWindow.Resize(fyne.NewSize(900, 650))         // Resize to final size
		simWindow.CenterOnScreen()
		log.Println("Main window content set and configured.")

		// Set close handler
		simWindow.SetOnClosed(func() {
			log.Println("Main window closed by user.")
			sim.cancel() // Signal simulation to stop
			waitTimeout := 2 * time.Second
			done := make(chan struct{})
			go func() {
				defer func() { recover(); close(done) }()
				sim.wg.Wait()
			}()
			select {
			case <-done:
				log.Println("Simulation finished cleanly.")
			case <-time.After(waitTimeout):
				log.Println("Warning: Timeout waiting for sim.")
			}
			log.Println("Exiting application.")
		})
	})

	// 9. Start the main event loop.
	log.Println("Starting main event loop...")
	myApp.Run()
	log.Println("Application finished.")
}

// showConfigDialog displays a modal dialog to get simulation parameters from the user.
// It blocks until the dialog is dismissed (OK or Cancel).
// Returns the parameters if OK was pressed, otherwise nil and false.
func showConfigDialog(a fyne.App, parent fyne.Window, defaults SimParams, onComplete func(*SimParams, bool)) {
	log.Println("Entering showConfigDialog...")

	// --- Widget Creation ---
	nxEntry := widget.NewEntry()
	nxEntry.SetText(strconv.Itoa(defaults.Nx))
	nxEntry.Validator = func(s string) error {
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			return errors.New("Nx must be a positive integer")
		}
		return nil
	}

	nyEntry := widget.NewEntry()
	nyEntry.SetText(strconv.Itoa(defaults.Ny))
	nyEntry.Validator = func(s string) error {
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			return errors.New("Ny must be a positive integer")
		}
		return nil
	}

	nzEntry := widget.NewEntry()
	nzEntry.SetText(strconv.Itoa(defaults.Nz))
	nzEntry.Validator = func(s string) error {
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			return errors.New("Nz must be a positive integer")
		}
		return nil
	}

	lEntry := widget.NewEntry()
	lEntry.SetText(fmt.Sprintf("%.2f", defaults.L))
	lEntry.Validator = func(s string) error {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil || f <= 0 {
			return errors.New("L must be a positive number")
		}
		return nil
	}

	omegaXEntry := widget.NewEntry()
	omegaXEntry.SetText(fmt.Sprintf("%.2f", defaults.OmegaX))
	omegaXEntry.Validator = func(s string) error {
		_, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return errors.New("ωx must be a number")
		}
		return nil
	}

	omegaYEntry := widget.NewEntry()
	omegaYEntry.SetText(fmt.Sprintf("%.2f", defaults.OmegaY))
	omegaYEntry.Validator = func(s string) error {
		_, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return errors.New("ωy must be a number")
		}
		return nil
	}

	omegaZEntry := widget.NewEntry()
	omegaZEntry.SetText(fmt.Sprintf("%.2f", defaults.OmegaZ))
	omegaZEntry.Validator = func(s string) error {
		_, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return errors.New("ωz must be a number")
		}
		return nil
	}

	items := []*widget.FormItem{
		widget.NewFormItem("Grid Nx", nxEntry),
		widget.NewFormItem("Grid Ny", nyEntry),
		widget.NewFormItem("Grid Nz", nzEntry),
		widget.NewFormItem("Axis Length (L)", lEntry),
		widget.NewFormItem("Potential Freq ωx", omegaXEntry),
		widget.NewFormItem("Potential Freq ωy", omegaYEntry),
		widget.NewFormItem("Potential Freq ωz", omegaZEntry),
	}

	log.Println("Creating and showing configuration dialog...")
	dialog.ShowForm("Simulation Setup", "Start Simulation", "Cancel", items,
		func(ok bool) {
			log.Printf("Dialog callback triggered. OK pressed: %v", ok)
			if !ok {
				log.Println("Configuration cancelled by user.")
				onComplete(nil, false)
				return
			}

			// Parse inputs
			nx, _ := strconv.Atoi(nxEntry.Text)
			ny, _ := strconv.Atoi(nyEntry.Text)
			nz, _ := strconv.Atoi(nzEntry.Text)
			lVal, _ := strconv.ParseFloat(lEntry.Text, 64)
			ox, _ := strconv.ParseFloat(omegaXEntry.Text, 64)
			oy, _ := strconv.ParseFloat(omegaYEntry.Text, 64)
			oz, _ := strconv.ParseFloat(omegaZEntry.Text, 64)

			log.Println("All checks passed. Storing results.")
			onComplete(&SimParams{
				Nx: nx, Ny: ny, Nz: nz,
				L: lVal,
				OmegaX: ox, OmegaY: oy, OmegaZ: oz,
			}, true)
		}, parent)
}