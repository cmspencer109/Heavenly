#include <wx/wx.h>
#include <wx/wxprec.h>
#include <wx/splitter.h>

class HeavenlyApp : public wxApp {
    public:
        virtual bool OnInit();
};
 
class HeavenlyFrame : public wxFrame {
    public:
        HeavenlyFrame();
    
    private:
        void OnHello(wxCommandEvent& event);
        void OnExit(wxCommandEvent& event);
        void OnAbout(wxCommandEvent& event);
};
 
enum {
    ID_Hello = 1
};
 
wxIMPLEMENT_APP(HeavenlyApp);
 
bool HeavenlyApp::OnInit() {
    HeavenlyFrame *frame = new HeavenlyFrame();
    frame->Show(true);
    return true;
}
 
HeavenlyFrame::HeavenlyFrame() : wxFrame(NULL, wxID_ANY, "Heavenly", wxDefaultPosition, wxSize(800, 600)) {
    // wxMenu *menuFile = new wxMenu;
    // menuFile->Append(ID_Hello, "&Hello...\tCtrl-H",
    //                  "Help string shown in status bar for this menu item");
    // menuFile->AppendSeparator();
    // menuFile->Append(wxID_EXIT);
 
    // wxMenu *menuHelp = new wxMenu;
    // menuHelp->Append(wxID_ABOUT);
 
    // wxMenuBar *menuBar = new wxMenuBar;
    // menuBar->Append(menuFile, "&File");
    // menuBar->Append(menuHelp, "&Help");
 
    // SetMenuBar( menuBar );
 
    // CreateStatusBar();
    // SetStatusText("Welcome to wxWidgets!");
 
    // Bind(wxEVT_MENU, &HeavenlyFrame::OnHello, this, ID_Hello);
    // Bind(wxEVT_MENU, &HeavenlyFrame::OnAbout, this, wxID_ABOUT);
    // Bind(wxEVT_MENU, &HeavenlyFrame::OnExit, this, wxID_EXIT);

    // // Centre(); // Center the frame
    // Maximize(); // Maximize the frame

    SetBackgroundColour(wxColour(50, 50, 50));

    // Create a wxSplitterWindow as the main window container
    wxSplitterWindow* splitter = new wxSplitterWindow(this, wxID_ANY);

    // Create a panel for the sidebar
    wxPanel* sidebarPanel = new wxPanel(splitter, wxID_ANY);
    sidebarPanel->SetBackgroundColour(wxColour(50, 50, 50)); // Set dark background color

    // Create a panel for the main content area
    wxPanel* contentPanel = new wxPanel(splitter, wxID_ANY);
    contentPanel->SetBackgroundColour(wxColour(60, 60, 60)); // Set slightly lighter background color

    // Set the minimum pane size for the sidebar
    splitter->SetMinimumPaneSize(100);

    // Split the window vertically, with the sidebar on the left
    splitter->SplitVertically(sidebarPanel, contentPanel);

    // Set the main sizer for the frame
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->Add(splitter, 1, wxEXPAND);
    SetSizer(mainSizer);

    // Add controls or content to the sidebar panel
    // For example:
    wxStaticText* sidebarText = new wxStaticText(sidebarPanel, wxID_ANY, "Sidebar Content");
    sidebarText->SetForegroundColour(wxColour(240, 240, 240)); // Set light text color

    // Add controls or content to the main content panel
    // For example:
    wxStaticText* mainContentText = new wxStaticText(contentPanel, wxID_ANY, "Main Content");
    mainContentText->SetForegroundColour(wxColour(240, 240, 240)); // Set light text color

    // Set the initial size of the sidebar panel (adjust as needed)
    sidebarPanel->SetMinSize(wxSize(200, -1));
}
 
void HeavenlyFrame::OnExit(wxCommandEvent& event) {
    Close(true);
}
 
void HeavenlyFrame::OnAbout(wxCommandEvent& event) {
    wxMessageBox("This is a wxWidgets Hello World example",
                 "About Hello World", wxOK | wxICON_INFORMATION);
}
 
void HeavenlyFrame::OnHello(wxCommandEvent& event) {
    wxLogMessage("Hello world from wxWidgets!");
}
