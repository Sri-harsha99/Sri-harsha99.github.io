import { BrowserModule } from '@angular/platform-browser';
import { A11yModule } from "@angular/cdk/a11y";
import { CommonModule } from "@angular/common";
import { CUSTOM_ELEMENTS_SCHEMA, NgModule } from "@angular/core";
import { FlexLayoutModule } from "@angular/flex-layout";
import { FormsModule, ReactiveFormsModule } from "@angular/forms";
import { MatButtonModule } from "@angular/material/button";
import { MatCardModule } from "@angular/material/card";
import { MatCheckboxModule } from "@angular/material/checkbox";
import { MatDatepickerModule } from "@angular/material/datepicker";
import { MatExpansionModule } from "@angular/material/expansion";
import { MatFormField, MatFormFieldModule } from "@angular/material/form-field";
import { MatDialogModule } from "@angular/material/dialog";
import { MatIconModule } from "@angular/material/icon";
import { MatInputModule } from "@angular/material/input";
import { MatListModule } from "@angular/material/list";
import { MatMenuModule } from "@angular/material/menu";
import { MatRadioModule } from "@angular/material/radio";
import { MatSelectModule } from "@angular/material/select";
import { MatSidenavModule } from "@angular/material/sidenav";
import { MatTabsModule } from "@angular/material/tabs";
import { MatToolbarModule } from "@angular/material/toolbar";
import { MatTooltipModule } from "@angular/material/tooltip";
import { MatTreeModule } from "@angular/material/tree";
import { MatChip, MatChipsModule, MatChipList } from '@angular/material/chips';
import { MatSlideToggleModule } from "@angular/material/slide-toggle";
import { MatProgressBarModule } from "@angular/material/progress-bar";
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner";
import { MatGridListModule } from "@angular/material/grid-list";
import { MatBottomSheetModule } from "@angular/material/bottom-sheet";
import { MatAutocompleteModule } from "@angular/material/autocomplete";
import { NgChartsModule } from 'ng2-charts';
import { MatSliderModule } from "@angular/material/slider";
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {BrowserAnimationsModule} from 
    '@angular/platform-browser/animations';
import { HomeComponent } from './home/home.component';
import { HttpClientModule } from '@angular/common/http';
import { MatNativeDateModule, MAT_DATE_LOCALE } from "@angular/material/core";
import {NgxMaterialTimepickerModule} from 'ngx-material-timepicker';
import { SpinnersAngularModule } from 'spinners-angular';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MzdTimelineModule } from 'ngx-mzd-timeline';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { MiniPlayerComponent } from './mini-player/mini-player.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    MiniPlayerComponent
  ],
  imports: [
    BrowserModule,
    NgChartsModule,
    AppRoutingModule,
    FontAwesomeModule,
    MatChipsModule,
    BrowserAnimationsModule,
    MatToolbarModule,
    MatIconModule,
    SpinnersAngularModule,
    MatInputModule,
    MatNativeDateModule,
    MatFormFieldModule,
    NgxMaterialTimepickerModule,
    CommonModule,
    FlexLayoutModule,
    MatSelectModule,
    MatListModule,
    HttpClientModule,
    MatButtonModule,
    MatTabsModule,
    MatDatepickerModule,
    MatFormFieldModule,
    MatExpansionModule,
    MatCheckboxModule,
    MatIconModule,
    FormsModule,
    MatMenuModule,
    ReactiveFormsModule,
    MatTreeModule,
    MatPaginatorModule,
    MatAutocompleteModule,
    MzdTimelineModule,
    MatFormFieldModule,
    MatInputModule,
    FormsModule,
    A11yModule,
    MatCardModule,
    MatTooltipModule,
    MatToolbarModule,
    MatRadioModule,
    MatSidenavModule,
    MatSlideToggleModule,
    MatDialogModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatGridListModule,
    MatBottomSheetModule,
    MatProgressSpinnerModule,
    MatSliderModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
