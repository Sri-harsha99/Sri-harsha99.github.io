import { Component, Input, Output,EventEmitter, OnInit, OnDestroy } from '@angular/core';

@Component({
  selector: 'app-typing-animation',
  template: `<span>{{ displayedText }}<span *ngIf="showCursor" class="cursor">|</span></span>`,
  styleUrls: ['./typing-animation.component.scss']
})
export class TypingAnimationComponent implements OnInit, OnDestroy {
  @Input() textToType: string = '';
  @Output() animationDone: EventEmitter<boolean> = new EventEmitter<boolean>();
  displayedText: string = '';
  currentIndex: number = 0;
  typingInterval: any;
  @Input() showCursor: boolean; // Control whether cursor should be displayed

  ngOnInit(): void {
    this.startTyping();
  }

  private startTyping() {
    this.typingInterval = setInterval(() => {
      if (this.currentIndex <= this.textToType.length) {
        this.displayedText = this.textToType.slice(0, this.currentIndex);
        this.currentIndex++;
      } else {
        if(!this.showCursor){
          clearInterval(this.typingInterval);
        } 
        this.checkAllAnimationsDone();
      }
    }, 100); // Adjust typing speed here (milliseconds)
  }

  private checkAllAnimationsDone() {
    this.animationDone.emit(true);
  }

  ngOnDestroy(): void {
    clearInterval(this.typingInterval);
  }
}
