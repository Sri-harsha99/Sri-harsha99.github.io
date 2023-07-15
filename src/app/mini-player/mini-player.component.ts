import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
@Component({
  selector: 'app-mini-player',
  templateUrl: './mini-player.component.html',
  styleUrls: ['./mini-player.component.scss']
})
export class MiniPlayerComponent implements OnInit {
  videoURL = 'https://youtube.com/embed/yvhQkRc0jKg';
  safeURL: any;
  constructor(private _sanitizer: DomSanitizer) {
    this.safeURL = this._sanitizer.bypassSecurityTrustResourceUrl(this.videoURL);
   }

  ngOnInit(): void {
  }
}
